"""``lean_constrained_beam_search``: segment-level beam search with delayed
(step-boundary-only) policy evaluation.

Implements the "token beam search with delayed policy evaluation" design from
``update.md``: the base model does ordinary, unconstrained token-level
generation -- ``<step>`` is a *decoder-level segmentation event*, never a
behavioural instruction to the model ("write ONE step, then stop"), the way
every other mode in ``guided_decoder.py`` prompts it. The trained classifier
fires *only* when a beam's generated text reaches a ``<step>`` boundary,
contributing one scalar to that beam's cumulative score; beams are then
pruned back to ``beam_width`` using the classifier-adjusted score, exactly as
in ordinary beam search. Between boundaries, only the base model's own
token-level probability drives which continuations survive -- the classifier
never touches an individual token (that is ``MODE_TOKEN``, a different,
already-existing mode this one is deliberately not built on).

Design note on the implementation strategy
-------------------------------------------
A single call to ``model.generate(num_beams=...)`` cannot pause mid-search to
inject an external score the instant *some* (but not all) beams cross a
``<step>`` boundary -- HuggingFace's beam search has no such hook. This
module instead runs beam search in ROUNDS, one round per reasoning step:

* Each of the (up to ``beam_width``) *surviving* beams from the previous
  round is a separate starting prefix.
* From each surviving beam, one ``model.generate(num_beams=expansion_per_beam,
  num_return_sequences=expansion_per_beam, stop_strings=["<step>"])`` call
  explores ``expansion_per_beam`` continuations *of that beam specifically*,
  using HuggingFace's own token-level beam search internally -- this is
  exactly the "usual" continuous, LM-driven pruning the design calls for,
  happening *within* a step, not reimplemented by hand.
* All beams' children are pooled (``n_survivors * expansion_per_beam``
  candidates), each newly-completed step is scored by the configured
  intervention (classifier logit, real Lean verification, or a hybrid of the
  two -- see ``BeamIntervention``), the score is added to that beam's running
  total, and the pool is pruned back to ``beam_width`` by the
  classifier-adjusted score. That pruning event is the "delayed policy
  evaluation" boundary the design describes.
* The first round has exactly one starting beam (the empty history), so it
  requests ``beam_width`` children directly rather than ``expansion_per_beam``
  per (nonexistent) sibling.

This is algorithmically faithful to the design -- real token-level beam
search within a step, classifier intervention only at boundaries, pruning
driven by the classifier-adjusted score -- while reusing HuggingFace's own,
battle-tested beam-search internals for the token-level part rather than
reimplementing padding, EOS handling, and length penalties by hand.

See ``lean-constrained-beam-search.md`` for the full design writeup,
including what this trades off against a single continuous token loop, and
what has and has not been verified against a live model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..data.trajectory import Trajectory
from ..protocol import contains_final_answer, strip_step_token
from ..state import PolicyState
from .guided_decoder import MAX_STEPS_SAFETY_CEILING, DecodeResult, DecodeStats, _make_record

MODE_LEAN_BEAM = "lean_constrained_beam_search"

#: What fires at each `<step>` boundary.
INTERVENE_CLASSIFIER = "classifier"
INTERVENE_VERIFIER = "verifier"
INTERVENE_HYBRID = "hybrid"


@dataclass
class BeamSearchConfig:
    beam_width: int = 4
    #: Candidate continuations explored per surviving beam at each round, via
    #: that beam's own internal (real, token-level) mini beam search. Pooling
    #: `beam_width * expansion_per_beam` candidates before pruning back to
    #: `beam_width` is what gives the classifier genuine pruning power at the
    #: boundary, rather than merely re-ordering who continues.
    expansion_per_beam: int = 2
    #: Classifier contribution weight -- same knob as `DecodeConfig.alpha`.
    alpha: float = 1.0
    #: HuggingFace's own GNMT-style length normalisation of the *base
    #: model's* score within a step (`update.md`'s beta): 0 = raw summed
    #: log-prob, 1 = roughly mean token log-prob. Never applied to the
    #: classifier's contribution -- point 1 of the request this implements:
    #: no length normalisation, and no distributing the classifier score
    #: across tokens (that is `MODE_TOKEN`, a setting that already exists).
    length_penalty: float = 1.0
    #: Raised from an original 128 after the live smoke test
    #: (`lean-constrained-beam-search.md` §7) observed a final-answer step
    #: truncated mid-expression at that value.
    max_tokens_per_step: int = 256
    #: `None` = no artificial cap, bounded only by `MAX_STEPS_SAFETY_CEILING`
    #: (see `guided_decoder.py`) and by every beam actually finishing. The
    #: live smoke test found the old fixed default (8, inherited from
    #: `DecodeConfig`) binding harder for a step-native model's finer step
    #: granularity than for the old `<step>`-forced one -- see
    #: `step-native-finetuning.md` §5/§6.
    max_steps: Optional[int] = None
    #: `INTERVENE_CLASSIFIER` (default), `INTERVENE_VERIFIER`, or
    #: `INTERVENE_HYBRID`.
    intervention: str = INTERVENE_CLASSIFIER
    #: Hybrid mode only, mirrors `DecodeConfig.hybrid_threshold`: escalate to
    #: a real Lean call only when the classifier's calibrated confidence
    #: (`predict_step_probability`, the *classification*-calibrated scaler,
    #: not `beam_calibrator`) falls below this -- Lean calls are the
    #: expensive resource, same rationale as `VERIFY_HYBRID` in
    #: `guided_decoder.py`.
    hybrid_threshold: float = 0.5
    #: Verifier-mode / hybrid-escalation score for a step that fails real
    #: Lean verification -- large enough that it is pruned out whenever any
    #: verified sibling survives, but finite so an all-failing pool still
    #: ranks by something instead of every beam tying at -inf.
    verify_reject_penalty: float = -50.0
    verify_accept_bonus: float = 0.0
    #: How a beam's per-step classifier contributions combine into its score:
    #: `"sum"` (the original design) or `"mean"` over the beam's scored steps.
    #: With "sum", the sign of the calibrated logit becomes a length preference --
    #: positive per-step scores make an unfinished beam outscore every finished one
    #: (finished beams stop accumulating), so the search runs to the step ceiling;
    #: negative ones favour the shortest solution. "mean" ranks by the average
    #: step score, which is what the true-formalization-rate metric measures.
    classifier_aggregate: str = "sum"
    seed: int = 0


@dataclass
class _Beam:
    """One surviving hypothesis.

    `lm_score` and `classifier_score` are tracked separately (both feed
    `.score`, which is what beams are ranked and pruned by) so a caller can
    decompose the final trajectory's score into "how much did the base model
    like this" vs. "how much did the policy like this" -- useful for the
    kind of analysis this whole investigation has been doing.
    """

    history: List[str] = field(default_factory=list)
    lm_score: float = 0.0
    classifier_score: float = 0.0
    #: Steps that contributed to `classifier_score` (for `classifier_aggregate="mean"`).
    n_scored: int = 0
    mean_aggregate: bool = False
    final_answer: str = ""
    finished: bool = False

    @property
    def score(self) -> float:
        if self.mean_aggregate:
            return self.lm_score + self.classifier_score / max(self.n_scored, 1)
        return self.lm_score + self.classifier_score

    def add_step_score(self, value: float) -> None:
        self.classifier_score += value
        self.n_scored += 1

    def child(self, step_text: str, lm_score: float) -> "_Beam":
        return _Beam(
            history=[*self.history, step_text],
            lm_score=self.lm_score + lm_score,
            classifier_score=self.classifier_score,
            n_scored=self.n_scored,
            mean_aggregate=self.mean_aggregate,
        )


def lean_constrained_beam_search(
    question: str,
    reference: Any,
    base_model: Any,
    policy: Any,
    verifier: Any = None,
    outcome_checker: Any = None,
    config: Optional[BeamSearchConfig] = None,
) -> DecodeResult:
    """`MODE_LEAN_BEAM`'s entry point -- see the module docstring for the design.

    `base_model` must implement `generate_step_beam` (added to
    `BaseModelAdapter`/`HFBaseModel` in `base_model.py` alongside
    `generate_step_candidates`); `policy` must implement
    `predict_step_logit_for_beam` (`INTERVENE_CLASSIFIER`/`INTERVENE_HYBRID`)
    and/or be paired with a `verifier` implementing `verify_many`
    (`INTERVENE_VERIFIER`/`INTERVENE_HYBRID`).
    """
    config = config or BeamSearchConfig()
    stats = DecodeStats()
    started = time.perf_counter()
    effective_max_steps = (
        config.max_steps if config.max_steps is not None else MAX_STEPS_SAFETY_CEILING
    )

    if config.classifier_aggregate not in ("sum", "mean"):
        raise ValueError(f"unknown classifier_aggregate: {config.classifier_aggregate!r}")
    beams: List[_Beam] = [_Beam(mean_aggregate=config.classifier_aggregate == "mean")]
    for step_index in range(effective_max_steps):
        stats.steps_attempted += 1
        active = [b for b in beams if not b.finished]
        if not active:
            break

        pool: List[_Beam] = list(_carry_over_finished(beams))
        for parent in active:
            n_children = config.beam_width if step_index == 0 else config.expansion_per_beam
            children = base_model.generate_step_beam(
                question,
                parent.history,
                num_beams=n_children,
                num_return_sequences=n_children,
                length_penalty=config.length_penalty,
                max_new_tokens=config.max_tokens_per_step,
                seed=config.seed,
            )
            for text, lm_score in children:
                stripped = strip_step_token(text)
                if not stripped:
                    continue
                stats.generated_tokens += len(stripped.split())
                child = parent.child(stripped, lm_score)
                if contains_final_answer(stripped):
                    # The final answer is not a reasoning step: it never goes
                    # to the classifier/verifier, matching `_next_step_candidates`
                    # in guided_decoder.py, and it is not appended to history.
                    child.finished = True
                    child.final_answer = stripped
                    child.history = list(parent.history)
                pool.append(child)

        if not pool:
            break  # nothing generated this round; keep the previous beams

        _apply_intervention(pool, question, policy, verifier, config, stats)
        beams = _prune(pool, config.beam_width)

        if all(b.finished for b in beams):
            break

    best = max(beams, key=lambda b: b.score) if beams else _Beam()
    trajectory = Trajectory(
        question=question,
        steps=[_make_record(s, None) for s in best.history],
        final_answer=best.final_answer,
        reference=None if reference is None else str(reference),
        metadata={
            "mode": MODE_LEAN_BEAM,
            "intervention": config.intervention,
            "beam_width": config.beam_width,
            "alpha": config.alpha,
            "final_score": best.score,
            "final_lm_score": best.lm_score,
            "final_classifier_score": best.classifier_score,
            "classifier_aggregate": config.classifier_aggregate,
        },
    )
    if outcome_checker is not None and reference is not None:
        outcome = outcome_checker.check(question, best.final_answer, reference)
        trajectory.outcome_reward = outcome.reward
        trajectory.metadata["outcome"] = outcome.to_dict()

    stats.latency_s = time.perf_counter() - started
    trajectory.metadata.update(stats.to_dict())
    return DecodeResult(trajectory=trajectory, stats=stats)


def _carry_over_finished(beams: List[_Beam]) -> List[_Beam]:
    """Beams that already finished on an earlier round: no new step to
    generate or score, they simply persist (competing on their frozen score)
    until every beam is finished or `max_steps` is hit."""
    return [b for b in beams if b.finished]


def _apply_intervention(
    pool: List[_Beam], question: str, policy: Any, verifier: Any,
    config: BeamSearchConfig, stats: DecodeStats,
) -> None:
    """Add each newly-completed (this round) beam's step contribution to
    `.classifier_score`, in place. Beams carried over already-finished are
    left untouched -- their contribution was applied when they finished."""
    candidates = [b for b in pool if not b.finished and b.history]
    if not candidates:
        return
    if config.intervention == INTERVENE_CLASSIFIER:
        for beam in candidates:
            beam.add_step_score(config.alpha * _beam_logit(policy, question, beam))
    elif config.intervention == INTERVENE_VERIFIER:
        _apply_verifier(candidates, question, verifier, config, stats)
    elif config.intervention == INTERVENE_HYBRID:
        _apply_hybrid(candidates, question, policy, verifier, config, stats)
    else:
        raise ValueError(f"unknown intervention: {config.intervention!r}")


def _beam_logit(policy: Any, question: str, beam: _Beam) -> float:
    state = PolicyState(question, beam.history[:-1], beam.history[-1])
    return policy.predict_step_logit_for_beam(state)


def _apply_verifier(
    candidates: List[_Beam], question: str, verifier: Any,
    config: BeamSearchConfig, stats: DecodeStats,
) -> None:
    if verifier is None:
        raise ValueError("intervention requires a verifier but none was given")
    # Group by shared parent history so siblings from the same expansion
    # share one `verify_many` call, the same batching `_rejection_sample`
    # uses in `guided_decoder.py` -- Lean calls are the expensive resource.
    by_history: Dict[tuple, List[_Beam]] = {}
    for beam in candidates:
        by_history.setdefault(tuple(beam.history[:-1]), []).append(beam)
    for history, group in by_history.items():
        steps = [b.history[-1] for b in group]
        results = verifier.verify_many(question, list(history), steps)
        stats.lean_calls += len(steps)
        for beam, result in zip(group, results):
            beam.add_step_score(
                config.verify_accept_bonus if result.verified else config.verify_reject_penalty
            )


def _apply_hybrid(
    candidates: List[_Beam], question: str, policy: Any, verifier: Any,
    config: BeamSearchConfig, stats: DecodeStats,
) -> None:
    """Trust the classifier when it's confident; escalate to real Lean
    verification only for the steps it's unsure about -- mirrors
    `guided_decoder._maybe_verify_online`'s `VERIFY_HYBRID` branch, adapted
    from "should we bother recording a verification" to "which contribution
    should this beam's score get." A beam gets exactly one of the two
    contributions, never both -- they are on unrelated scales."""
    unsure: List[_Beam] = []
    for beam in candidates:
        state = PolicyState(question, beam.history[:-1], beam.history[-1])
        confidence = policy.predict_step_probability(state)
        if confidence >= config.hybrid_threshold:
            beam.add_step_score(config.alpha * policy.predict_step_logit_for_beam(state))
        else:
            unsure.append(beam)
    if unsure:
        _apply_verifier(unsure, question, verifier, config, stats)


def _prune(pool: List[_Beam], beam_width: int) -> List[_Beam]:
    return sorted(pool, key=lambda b: b.score, reverse=True)[:beam_width]
