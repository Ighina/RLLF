"""Guided decoding: the frozen base model generates, `pi_phi` steers.

Four decoding modes, matching the staged rollout the methodology recommends:

* ``plain``     -- baseline; the base model alone (equivalent to alpha = 0).
* ``rerank``    -- stage 2; sample K complete candidate steps, keep the one the
                   policy scores highest.
* ``rejection`` -- stage 3; rerank, then verify the top candidates in Lean and
                   prefer a verified one, resampling on failure.
* ``token``     -- stage 4; per-token guidance,
                   ``log p'(a) = log p_base(a) + alpha * log pi_phi(a | state)``.

Verification can be online (immediately at each `<step>`), deferred (after the
whole answer), hybrid (online only when the policy is unsure), or off.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from ..data.trajectory import StepRecord, Trajectory
from ..protocol import (
    contains_final_answer,
    contains_step_boundary,
    extract_final_answer,
    strip_step_token,
)
from ..policy.model import Action
from ..policy.scoring import PolicyAdapter, UniformPolicy, combine_scores, select_top_candidates
from ..state import PolicyState
from ..verification.base import SKIPPED, StepVerification

# Decoding modes.
MODE_PLAIN = "plain"
MODE_RERANK = "rerank"
MODE_REJECTION = "rejection"
MODE_TOKEN = "token"

# Verification schedules (section 10).
VERIFY_NONE = "none"
VERIFY_ONLINE = "online"
VERIFY_DEFERRED = "deferred"
VERIFY_HYBRID = "hybrid"

#: A hard ceiling on step count, applied only when `DecodeConfig.max_steps`
#: is `None` ("no artificial budget"). Found necessary after discovering
#: `max_steps=6/8` was silently zeroing out a *large* fraction of every
#: MATH-500 arm run so far -- for `MODE_PLAIN`/`MODE_RERANK`/`MODE_REJECTION`,
#: `force_finish` (below) is a `MODE_TOKEN`-only mechanism, so a trajectory
#: that simply hasn't reached a final answer by `max_steps` ends with
#: `final_answer == ""` -- a guaranteed-wrong outcome, not a genuine
#: incorrect-reasoning one. Measured on every existing MATH-500 run: 100% of
#: trajectories that hit the cap had an empty final answer, at cap rates of
#: 19-52% depending on the arm (rerank-family arms hit it roughly twice as
#: often as `self_consistency`) -- see `experiments_v1.md`'s confound note.
#: This ceiling exists only to stop a genuinely degenerate/non-terminating
#: generation from consuming unbounded compute; it is not meant to express
#: any reasoning-length policy, unlike the old fixed default was implicitly
#: doing.
MAX_STEPS_SAFETY_CEILING = 200


@dataclass
class DecodeConfig:
    mode: str = MODE_RERANK
    num_candidates: int = 4
    #: Guidance strength; alpha = 0 reproduces the base model exactly.
    alpha: float = 0.5
    #: `None` = no artificial cap, bounded only by `MAX_STEPS_SAFETY_CEILING`
    #: and by the model actually reaching a final answer -- see that
    #: constant's docstring for why this is no longer a fixed small default.
    #: Pass an explicit int to reproduce the old fixed-budget behaviour.
    max_steps: Optional[int] = None
    max_tokens_per_step: int = 128
    #: Safeguards against pathological boundary decisions (section 9).
    min_step_tokens: int = 4
    verification: str = VERIFY_DEFERRED
    #: In hybrid mode, verify online only when the predicted step probability
    #: falls below this threshold -- Lean calls are the expensive resource.
    hybrid_threshold: float = 0.5
    #: How many of the top-scoring candidates to send to Lean per step.
    verify_top_m: int = 2
    max_repairs: int = 2
    #: Token mode only.
    top_k: int = 16
    temperature: float = 1.0
    greedy: bool = True
    #: Let the policy's action head decide when to close a step.
    use_action_head: bool = False
    seed: int = 0


@dataclass
class DecodeStats:
    generated_tokens: int = 0
    lean_calls: int = 0
    candidates_scored: int = 0
    resamples: int = 0
    steps_attempted: int = 0
    latency_s: float = 0.0

    @property
    def resample_rate(self) -> float:
        return self.resamples / self.steps_attempted if self.steps_attempted else 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "generated_tokens": float(self.generated_tokens),
            "lean_calls": float(self.lean_calls),
            "candidates_scored": float(self.candidates_scored),
            "resamples": float(self.resamples),
            "steps_attempted": float(self.steps_attempted),
            "latency_s": self.latency_s,
            "resample_rate": self.resample_rate,
        }


@dataclass
class DecodeResult:
    trajectory: Trajectory
    stats: DecodeStats = field(default_factory=DecodeStats)

    @property
    def steps(self) -> List[str]:
        return self.trajectory.step_texts

    @property
    def final_answer(self) -> str:
        return self.trajectory.final_answer

    @property
    def outcome(self) -> float:
        return self.trajectory.outcome_reward


class GuidedDecoder:
    """Runs the guided generation loop for one question at a time."""

    def __init__(
        self,
        base_model: Any,
        policy: Optional[PolicyAdapter] = None,
        verifier: Any = None,
        outcome_checker: Any = None,
        config: Optional[DecodeConfig] = None,
    ):
        self.base_model = base_model
        self.policy = policy or UniformPolicy()
        self.verifier = verifier
        self.outcome_checker = outcome_checker
        self.config = config or DecodeConfig()

    # -- public API -------------------------------------------------------
    def generate(self, question: str, reference: Any = None) -> DecodeResult:
        config = self.config
        stats = DecodeStats()
        started = time.perf_counter()
        torch.manual_seed(config.seed)

        history: List[str] = []
        records: List[StepRecord] = []
        final_answer = ""
        effective_max_steps = (
            config.max_steps if config.max_steps is not None else MAX_STEPS_SAFETY_CEILING
        )

        for step_index in range(effective_max_steps):
            stats.steps_attempted += 1
            last_step = step_index == effective_max_steps - 1
            step_text, verification = self._next_step(
                question, history, stats, force_finish=last_step
            )
            if step_text is None:
                break
            if contains_final_answer(step_text):
                final_answer = step_text
                break
            records.append(_make_record(step_text, verification))
            history.append(step_text)

        # Deferred / hybrid top-up: everything not already checked online.
        if self.verifier is not None and config.verification in (VERIFY_DEFERRED, VERIFY_HYBRID):
            self._verify_deferred(question, records, stats)

        trajectory = Trajectory(
            question=question,
            steps=records,
            final_answer=final_answer,
            reference=None if reference is None else str(reference),
            metadata={"mode": config.mode, "alpha": config.alpha, "verification": config.verification},
        )
        if self.outcome_checker is not None and reference is not None:
            outcome = self.outcome_checker.check(question, final_answer, reference)
            trajectory.outcome_reward = outcome.reward
            trajectory.metadata["outcome"] = outcome.to_dict()

        stats.latency_s = time.perf_counter() - started
        trajectory.metadata.update(stats.to_dict())
        return DecodeResult(trajectory=trajectory, stats=stats)

    def generate_many(
        self, items: Sequence[Tuple[str, Any]], progress: bool = False
    ) -> List[DecodeResult]:
        results = []
        for index, (question, reference) in enumerate(items):
            if progress:
                print(f"[{index + 1}/{len(items)}] {question[:60]}")
            results.append(self.generate(question, reference))
        return results

    # -- one step ---------------------------------------------------------
    def _next_step(
        self, question: str, history: Sequence[str], stats: DecodeStats, force_finish: bool = False
    ) -> Tuple[Optional[str], Optional[StepVerification]]:
        if self.config.mode == MODE_TOKEN:
            return self._next_step_tokens(question, history, stats, force_finish), None
        return self._next_step_candidates(question, history, stats)

    def _next_step_candidates(
        self, question: str, history: Sequence[str], stats: DecodeStats
    ) -> Tuple[Optional[str], Optional[StepVerification]]:
        """Stage 2/3: sample complete steps, rerank, optionally verify."""
        config = self.config
        candidates = self._sample_candidates(question, history, stats)
        if not candidates:
            return None, None

        scores = self._score_candidates(question, history, candidates, stats)
        order = self._rank_candidates(scores)
        best = candidates[order[0]]

        # The final answer is not a reasoning step: it goes to the outcome
        # checker, never to the step verifier.
        if contains_final_answer(best):
            return best, None

        if config.mode in (MODE_PLAIN, MODE_RERANK) or self.verifier is None:
            verification = self._maybe_verify_online(question, history, best, scores[order[0]], stats)
            return best, verification

        # Stage 3: verification-aware rejection sampling.
        selected, verification = self._rejection_sample(
            question, history, candidates, scores, order, stats
        )
        return selected, verification

    def _sample_candidates(
        self, question: str, history: Sequence[str], stats: DecodeStats
    ) -> List[str]:
        n = 1 if self.config.mode == MODE_PLAIN else self.config.num_candidates
        # `do_sample`/`temperature` are the Hugging Face spelling, so the same
        # call works for an HF model, an API adapter, or a test double.
        candidates = list(
            self.base_model.generate_step_candidates(
                question,
                list(history),
                num_candidates=n,
                do_sample=not (self.config.greedy and n == 1),
                temperature=self.config.temperature,
            )
        )
        candidates = [c for c in (strip_step_token(c) for c in candidates) if c]
        stats.generated_tokens += sum(len(c.split()) for c in candidates)
        # Deduplicate while preserving order: identical candidates waste Lean calls.
        seen, unique = set(), []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique.append(candidate)
        return unique

    def _score_candidates(
        self, question: str, history: Sequence[str], candidates: Sequence[str], stats: DecodeStats
    ) -> torch.Tensor:
        if self.config.mode == MODE_PLAIN or self.config.alpha == 0:
            return torch.zeros(len(candidates))
        stats.candidates_scored += len(candidates)
        return self.policy.score_step_candidates(question, list(history), list(candidates))

    def _rank_candidates(self, scores: torch.Tensor) -> List[int]:
        """Order candidates by guided preference.

        `alpha` is the same knob as in the token-level rule: at 0 the base
        model's own first sample wins and the policy is ignored, and larger
        values sharpen the preference.  With `greedy=False` the choice is
        sampled from `softmax(alpha * scores)`, which makes alpha a continuous
        guidance strength at the candidate level rather than an on/off switch.
        """
        n = len(scores)
        if n <= 1 or self.config.alpha == 0:
            return list(range(n))
        weighted = torch.as_tensor(scores, dtype=torch.float) * self.config.alpha
        if self.config.greedy:
            return sorted(range(n), key=lambda i: float(weighted[i]), reverse=True)
        probabilities = torch.softmax(weighted / max(self.config.temperature, 1e-3), dim=-1)
        return torch.multinomial(probabilities, n).tolist()

    def _rejection_sample(
        self,
        question: str,
        history: Sequence[str],
        candidates: List[str],
        scores: torch.Tensor,
        order: List[int],
        stats: DecodeStats,
    ) -> Tuple[str, Optional[StepVerification]]:
        """Verify the most promising candidates; resample if none survive."""
        config = self.config
        pool = list(candidates)
        pool_order = list(order)
        best_overall = pool[pool_order[0]]

        for attempt in range(config.max_repairs + 1):
            probe = [pool[i] for i in pool_order[: max(1, config.verify_top_m)]]
            results = self.verifier.verify_many(question, list(history), probe)
            stats.lean_calls += len(probe)
            for candidate, result in zip(probe, results):
                if result.verified:
                    return candidate, result
            if attempt == config.max_repairs:
                # No candidate verified: keep the best-scoring one and record it.
                return best_overall, results[0] if results else None
            stats.resamples += 1
            pool = self._sample_candidates(question, history, stats)
            if not pool:
                return best_overall, None
            scores = self._score_candidates(question, history, pool, stats)
            pool_order = self._rank_candidates(scores)
            best_overall = pool[pool_order[0]]
        return best_overall, None

    def _maybe_verify_online(
        self,
        question: str,
        history: Sequence[str],
        step: str,
        score: Any,
        stats: DecodeStats,
    ) -> Optional[StepVerification]:
        config = self.config
        if self.verifier is None or config.verification == VERIFY_NONE:
            return None
        if config.verification == VERIFY_DEFERRED:
            return None
        if config.verification == VERIFY_HYBRID:
            confidence = _score_to_probability(self.policy, question, history, step, score)
            if confidence >= config.hybrid_threshold:
                return None  # confident enough to skip the Lean call
        stats.lean_calls += 1
        return self.verifier.verify(question, list(history), step)

    def _verify_deferred(
        self, question: str, records: List[StepRecord], stats: DecodeStats
    ) -> None:
        history: List[str] = []
        for record in records:
            if record.status in ("unverified", SKIPPED):
                result = self.verifier.verify(question, list(history), record.text)
                stats.lean_calls += 1
                _apply_verification(record, result)
            history.append(record.text)

    # -- token-level guidance (stage 4) -----------------------------------
    def _next_step_tokens(
        self, question: str, history: Sequence[str], stats: DecodeStats, force_finish: bool
    ) -> Optional[str]:
        config = self.config
        tokenizer = self.base_model.tokenizer
        current = ""
        n_tokens = 0

        for _ in range(config.max_tokens_per_step):
            state = PolicyState(question, list(history), current)
            base_logits = self.base_model.next_token_logits(question, list(history), current)
            if base_logits is None or base_logits.numel() == 0:
                break
            candidate_ids = select_top_candidates(base_logits, config.top_k)
            combined = self._combine(base_logits, candidate_ids, state, stats)
            choice = self._select(combined)
            token_id = int(candidate_ids[choice])
            piece = tokenizer.decode([token_id])
            if not piece:
                break
            current += piece
            n_tokens += 1
            stats.generated_tokens += 1

            action = self._boundary_action(state.with_step(current), n_tokens, force_finish)
            if action is Action.CLOSE_STEP or contains_step_boundary(current):
                break
            if action is Action.END_ANSWER or contains_final_answer(current):
                break

        current = strip_step_token(current)
        return current or None

    def _combine(
        self,
        base_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        state: PolicyState,
        stats: DecodeStats,
    ) -> torch.Tensor:
        base_log_probs = torch.log_softmax(base_logits, dim=-1)[candidate_ids]
        if self.config.alpha == 0 or not getattr(self.policy, "supports_token_scores", False):
            return base_log_probs
        policy_scores = self.policy.score_tokens(state, candidate_ids.tolist())
        stats.candidates_scored += len(candidate_ids)
        # `combine_scores` normalises both sides, so alpha means the same thing
        # regardless of how the policy scales its raw outputs.
        return combine_scores(base_log_probs, policy_scores, self.config.alpha)

    def _select(self, scores: torch.Tensor) -> int:
        if self.config.greedy:
            return int(scores.argmax())
        probabilities = torch.softmax(scores / max(self.config.temperature, 1e-3), dim=-1)
        return int(torch.multinomial(probabilities, 1))

    def _boundary_action(
        self, state: PolicyState, n_tokens: int, force_finish: bool
    ) -> Optional[Action]:
        """Learned boundary decision, wrapped in the section 9 safeguards.

        The safeguards dominate the learned choice: a step may not close before
        `min_step_tokens` nor run past `max_tokens_per_step`.  `force_finish`
        (we are on the last permitted step) turns a *close* into a *finish*; it
        never truncates a step that is still being written, which would emit
        the half-formed steps the safeguards exist to prevent.
        """
        config = self.config
        if n_tokens < config.min_step_tokens:
            return Action.CONTINUE_STEP
        if n_tokens >= config.max_tokens_per_step:
            return Action.END_ANSWER if force_finish else Action.CLOSE_STEP
        if not config.use_action_head:
            return None
        try:
            scores = self.policy.action_scores(state)
        except (NotImplementedError, RuntimeError):
            return None
        action = max(scores, key=scores.get)
        if action is Action.CLOSE_STEP and force_finish:
            return Action.END_ANSWER
        return action


# --------------------------------------------------------------------------
# Helpers and standalone routines
# --------------------------------------------------------------------------


def _make_record(step_text: str, verification: Optional[StepVerification]) -> StepRecord:
    record = StepRecord(text=step_text)
    if verification is not None:
        _apply_verification(record, verification)
    return record


def _apply_verification(record: StepRecord, result: StepVerification) -> None:
    record.formalization = result.formalization
    record.verified = result.verified
    record.reward = result.reward
    record.status = result.status
    record.error = result.error
    record.diagnostics = dict(result.diagnostics)


def _score_to_probability(
    policy: PolicyAdapter, question: str, history: Sequence[str], step: str, score: Any
) -> float:
    """Policy confidence that the step will verify, for the hybrid schedule."""
    predictor = getattr(policy, "predict_step_probability", None)
    if predictor is not None:
        return predictor(PolicyState(question, list(history), step))
    return float(torch.sigmoid(torch.tensor(float(score))))


def sample_verified_step(
    question: str,
    history: Sequence[str],
    generator: Any,
    verifier: Any,
    attempts: int = 4,
    policy: Optional[PolicyAdapter] = None,
) -> Tuple[str, Optional[StepVerification]]:
    """Rejection-sample a single step until one verifies (section 10).

    Falls back to the highest-scoring candidate -- ranked by the policy when
    one is supplied, otherwise by generation order -- and returns its failed
    verification so the caller can record the failure.
    """
    candidates = [
        strip_step_token(c)
        for c in generator.generate_step_candidates(question, list(history), num_candidates=attempts)
    ]
    candidates = [c for c in candidates if c]
    if not candidates:
        return "", None
    results = verifier.verify_many(question, list(history), candidates)
    for candidate, result in zip(candidates, results):
        if result.verified:
            return candidate, result
    if policy is not None:
        scores = policy.score_step_candidates(question, list(history), candidates)
        best = int(torch.as_tensor(scores).argmax())
    else:
        best = 0
    return candidates[best], results[best]


def self_consistency(
    question: str,
    reference: Any,
    base_model: Any,
    n: int = 4,
    config: Optional[DecodeConfig] = None,
    outcome_checker: Any = None,
    verifier: Any = None,
) -> DecodeResult:
    """Majority vote over `n` independently sampled solutions.

    The standard test-time-compute baseline, and the right control when the
    comparison already samples several continuations: it spends the same budget
    on whole trajectories and aggregates by *answer* rather than by step score.
    """
    from ..verification.outcome_checker import normalize_answer

    config = config or DecodeConfig(mode=MODE_PLAIN, greedy=False, verification=VERIFY_NONE)
    samples: List[DecodeResult] = []
    for index in range(n):
        sample_config = DecodeConfig(**{**config.__dict__, "seed": config.seed + index, "greedy": False})
        decoder = GuidedDecoder(
            base_model, UniformPolicy(), verifier, outcome_checker, sample_config
        )
        samples.append(decoder.generate(question, reference))

    votes: Dict[str, int] = {}
    for result in samples:
        key = normalize_answer(extract_final_answer(result.trajectory.final_answer))
        if key:
            votes[key] = votes.get(key, 0) + 1
    winner = max(votes, key=votes.get) if votes else None

    chosen = samples[0]
    if winner is not None:
        for result in samples:
            if normalize_answer(extract_final_answer(result.trajectory.final_answer)) == winner:
                chosen = result
                break
    chosen.trajectory.metadata.update(
        {"self_consistency_n": n, "votes": dict(votes), "winning_votes": votes.get(winner, 0)}
    )
    for field_name in ("generated_tokens", "lean_calls", "latency_s"):
        setattr(chosen.stats, field_name, sum(getattr(r.stats, field_name) for r in samples))
    chosen.trajectory.metadata.update(chosen.stats.to_dict())
    return chosen


def _weighted_majority(
    samples: Sequence[DecodeResult], weights: Sequence[float]
) -> Tuple[DecodeResult, Dict[str, float], Optional[str]]:
    """Pick the answer with the highest total weight, not the highest vote count.

    Returns `(chosen, vote_weight_by_answer, winning_answer)`. Among samples
    that reached the winning answer, `chosen` is the one `weights` trusts
    most -- consistent with weighting the vote by that same score. Pure and
    generation-free so it can be tested without a base model or policy.
    """
    from ..verification.outcome_checker import normalize_answer

    vote_weight: Dict[str, float] = {}
    for result, weight in zip(samples, weights):
        key = normalize_answer(extract_final_answer(result.trajectory.final_answer))
        if key:
            vote_weight[key] = vote_weight.get(key, 0.0) + weight
    winner = max(vote_weight, key=vote_weight.get) if vote_weight else None

    chosen, chosen_weight = samples[0], -1.0
    for result, weight in zip(samples, weights):
        if winner is None:
            break
        if normalize_answer(extract_final_answer(result.trajectory.final_answer)) == winner and weight > chosen_weight:
            chosen, chosen_weight = result, weight
    return chosen, vote_weight, winner


def self_consistency_weighted(
    question: str,
    reference: Any,
    base_model: Any,
    policy: PolicyAdapter,
    n: int = 4,
    config: Optional[DecodeConfig] = None,
    outcome_checker: Any = None,
    verifier: Any = None,
) -> DecodeResult:
    """Self-consistency where each sample's vote is the policy's reward, not 1.

    Same `n` independently-sampled whole solutions as `self_consistency`
    (generation itself still uses `UniformPolicy` -- the reward model never
    steers, only judges after the fact), but each candidate's vote for its
    answer is weighted by `score_trajectory_formalizability(policy, ...)`
    instead of counted as 1. An answer three weak trajectories agree on can
    lose to an answer one confidently-formalizable trajectory reaches alone.
    """
    from ..policy.scoring import score_trajectory_formalizability

    config = config or DecodeConfig(mode=MODE_PLAIN, greedy=False, verification=VERIFY_NONE)
    samples: List[DecodeResult] = []
    weights: List[float] = []
    for index in range(n):
        sample_config = DecodeConfig(**{**config.__dict__, "seed": config.seed + index, "greedy": False})
        decoder = GuidedDecoder(base_model, UniformPolicy(), verifier, outcome_checker, sample_config)
        result = decoder.generate(question, reference)
        samples.append(result)
        weights.append(score_trajectory_formalizability(policy, result.trajectory))

    chosen, vote_weight, winner = _weighted_majority(samples, weights)
    chosen.trajectory.metadata.update(
        {
            "self_consistency_n": n,
            "vote_weights": vote_weight,
            "winning_weight": vote_weight.get(winner, 0.0),
            "sample_weights": weights,
        }
    )
    for field_name in ("generated_tokens", "lean_calls", "latency_s"):
        setattr(chosen.stats, field_name, sum(getattr(r.stats, field_name) for r in samples))
    chosen.trajectory.metadata.update(chosen.stats.to_dict())
    return chosen


def full_trajectory_rerank(
    question: str,
    reference: Any,
    base_model: Any,
    policy: PolicyAdapter,
    n: int = 10,
    outcome_checker: Any = None,
    config: Optional[DecodeConfig] = None,
) -> DecodeResult:
    """Rerank `n` whole, naturally-generated solutions instead of `MODE_RERANK`'s
    `<step>`-token candidates.

    `rerank` forces the base model to write one short step, emit `<step>`, and
    stop -- a generation-time boundary nothing in FaithformBench's own step
    data ever used: those steps come from an unconstrained chain-of-thought
    that was segmented *after* generation, the ProcessBench convention. This
    tests whether that generation-protocol mismatch, not just which base model
    is decoding, explains why a classifier trained on naturally-segmented
    steps transfers poorly to `rerank`'s candidates (see `experiments_v1.md`
    §8's still-open easy-problem anomaly).

    Pipeline: (1) sample `n` complete solutions with
    `base_model.generate_full_solution` -- free prose, no `<step>` forcing;
    (2) ask the same base model to retroactively segment each one with
    `base_model.segment_into_steps` (`rllf/protocol.py`'s
    `parse_numbered_steps` reads the numbered-list reply back); (3) score
    each candidate's mean per-step formalizability under `policy`
    (`score_trajectory_formalizability`); (4) keep the highest-scoring one,
    exactly as `rerank` keeps its highest-scoring step, but at whole-trajectory
    granularity. Every candidate (not just the winner) is kept in
    `trajectory.metadata["all_candidates"]` so an offline pipeline can
    real-Lean check the full sampled population, not only the pick -- needed
    to compare the classifier's choice against the population's true rate,
    and to ablate `n` after the fact without resampling.
    """
    from ..policy.scoring import score_trajectory_formalizability

    config = config or DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_NONE)
    stats = DecodeStats()
    started = time.perf_counter()

    texts = base_model.generate_full_solution(
        question, num_candidates=n, temperature=config.temperature, seed=config.seed
    )
    stats.generated_tokens += sum(len(t.split()) for t in texts)
    stats.steps_attempted = len(texts)

    candidates: List[Dict[str, Any]] = []
    for text in texts:
        steps = base_model.segment_into_steps(text)
        final_answer = text if contains_final_answer(text) else ""
        trajectory = Trajectory(
            question=question,
            steps=[_make_record(s, None) for s in steps],
            final_answer=final_answer,
        )
        stats.candidates_scored += len(steps)
        score = score_trajectory_formalizability(policy, trajectory) if steps else 0.0
        candidates.append(
            {"text": text, "steps": steps, "final_answer": final_answer, "score": score}
        )

    best = max(candidates, key=lambda c: c["score"]) if candidates else {
        "text": "", "steps": [], "final_answer": "", "score": 0.0,
    }
    trajectory = Trajectory(
        question=question,
        steps=[_make_record(s, None) for s in best["steps"]],
        final_answer=best["final_answer"],
        reference=None if reference is None else str(reference),
        metadata={
            "mode": "full_trajectory_rerank",
            "n": n,
            "chosen_score": best["score"],
            "all_candidates": candidates,
        },
    )
    if outcome_checker is not None and reference is not None:
        outcome = outcome_checker.check(question, best["final_answer"], reference)
        trajectory.outcome_reward = outcome.reward
        trajectory.metadata["outcome"] = outcome.to_dict()

    stats.latency_s = time.perf_counter() - started
    trajectory.metadata.update(stats.to_dict())
    return DecodeResult(trajectory=trajectory, stats=stats)


def best_of_n(
    question: str,
    reference: Any,
    base_model: Any,
    scorer: Optional[PolicyAdapter],
    n: int = 8,
    config: Optional[DecodeConfig] = None,
    verifier: Any = None,
    outcome_checker: Any = None,
    rank_by: str = "policy",
) -> DecodeResult:
    """Baseline 3/4: sample N complete trajectories and keep the best one.

    `rank_by="policy"` uses the reward model's final-answer score;
    `rank_by="verification"` uses the fraction of steps Lean accepts, which is
    the verification-only rejection-sampling baseline.
    """
    config = config or DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_NONE)
    results: List[DecodeResult] = []
    for index in range(n):
        sample_config = DecodeConfig(**{**config.__dict__, "seed": config.seed + index})
        decoder = GuidedDecoder(
            base_model,
            policy=scorer or UniformPolicy(),
            verifier=verifier,
            outcome_checker=outcome_checker,
            config=sample_config,
        )
        results.append(decoder.generate(question, reference))

    def key(result: DecodeResult) -> float:
        if rank_by == "verification":
            steps = result.trajectory.steps
            return sum(s.reward for s in steps) / len(steps) if steps else 0.0
        if scorer is None:
            return 0.0
        return scorer.score_final_answer(result.trajectory)

    best = max(results, key=key)
    best.trajectory.metadata["best_of_n"] = n
    best.stats.generated_tokens = sum(r.stats.generated_tokens for r in results)
    best.stats.lean_calls = sum(r.stats.lean_calls for r in results)
    best.stats.latency_s = sum(r.stats.latency_s for r in results)
    best.trajectory.metadata.update(best.stats.to_dict())
    return best
