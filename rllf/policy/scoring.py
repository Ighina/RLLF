"""Scoring interfaces, score combination, and calibration.

`pi_phi` never generates text.  It scores what the frozen base model proposes,
at three granularities:

* whole candidate steps (reranking -- stage 2, the recommended starting point);
* individual next tokens (logit guidance -- stage 4);
* step-boundary actions (continue / close step / finish).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from ..data.trajectory import Trajectory
from ..state import PolicyState, render
from .model import ACTIONS, Action, ProcessRewardModel


class PolicyAdapter:
    """Interface the guided decoder programs against."""

    #: True if `score_tokens` returns meaningful per-token scores.
    supports_token_scores: bool = False

    def score_step_completion(self, state: PolicyState) -> float:
        """Score for completing the current step (higher is better)."""
        raise NotImplementedError

    def score_step_candidates(
        self, question: str, history: Sequence[str], candidates: Sequence[str]
    ) -> torch.Tensor:
        """One score per complete candidate step."""
        return torch.tensor(
            [
                self.score_step_completion(PolicyState(question, list(history), candidate))
                for candidate in candidates
            ]
        )

    def score_tokens(self, state: PolicyState, token_ids: Sequence[int]) -> torch.Tensor:
        """One score per candidate next token."""
        raise NotImplementedError

    def score_final_answer(self, trajectory: Trajectory | PolicyState) -> float:
        """Optional score for the completed answer."""
        raise NotImplementedError

    def action_scores(self, state: PolicyState) -> Dict[Action, float]:
        """Scores for the step-boundary decision."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Calibration (section 14: a reward logit is not a log-probability)
# --------------------------------------------------------------------------


@dataclass
class TemperatureScaler:
    """Platt scaling `sigmoid((z - b) / T)` fitted on held-out states."""

    temperature: float = 1.0
    bias: float = 0.0
    #: Bounds keep a degenerate fit (e.g. labels uncorrelated with the score)
    #: from collapsing the policy into a constant.
    min_temperature: float = 0.05
    max_temperature: float = 20.0

    def apply_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return (logits - self.bias) / max(self.temperature, 1e-3)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.apply_logits(logits))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, steps: int = 200) -> "TemperatureScaler":
        logits = logits.detach().float().flatten()
        labels = labels.detach().float().flatten()
        if logits.numel() == 0 or labels.unique().numel() < 2:
            return self  # nothing to calibrate against
        log_t = torch.zeros(1, requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t, bias], lr=0.1, max_iter=steps)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_fn((logits - bias) / log_t.exp(), labels)
            loss.backward()
            return loss

        optimizer.step(closure)  # type: ignore[arg-type]
        temperature = float(log_t.exp().item())
        if not (temperature == temperature):  # NaN
            return self
        self.temperature = min(max(temperature, self.min_temperature), self.max_temperature)
        self.bias = float(bias.item())
        return self

    def to_dict(self) -> Dict[str, float]:
        return {"temperature": self.temperature, "bias": self.bias}


# --------------------------------------------------------------------------
# The learned policy
# --------------------------------------------------------------------------


class ProcessRewardPolicy(PolicyAdapter):
    """Wraps a trained :class:`ProcessRewardModel` as a decoding policy.

    The scalar handed to the decoder is

        score = log p(step verifies) + outcome_weight * log p(answer correct)

    i.e. a log-probability, which is the only form that may be added to base
    model log-probabilities without an arbitrary scale.
    """

    def __init__(
        self,
        model: ProcessRewardModel,
        outcome_weight: float = 0.5,
        step_calibrator: Optional[TemperatureScaler] = None,
        outcome_calibrator: Optional[TemperatureScaler] = None,
        token_scoring: str = "lookahead",
        base_tokenizer: Any = None,
        batch_size: int = 32,
    ):
        self.model = model
        self.model.eval()
        self.outcome_weight = outcome_weight
        self.step_calibrator = step_calibrator or TemperatureScaler()
        self.outcome_calibrator = outcome_calibrator or TemperatureScaler()
        # "head": use the policy's own token head (requires shared tokenization
        # with the base model).  "lookahead": score the state that each
        # candidate token would produce -- tokenizer-independent, and the only
        # option when the policy and base model do not share a vocabulary.
        self.token_scoring = token_scoring
        self.base_tokenizer = base_tokenizer
        self.batch_size = batch_size

    @property
    def supports_token_scores(self) -> bool:
        if self.token_scoring == "head":
            return self.model.token_head is not None
        return self.base_tokenizer is not None

    # -- core scoring -----------------------------------------------------
    def _score_texts(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        chunks: List[Dict[str, torch.Tensor]] = []
        for start in range(0, len(texts), self.batch_size):
            chunks.append(self.model.score_texts(texts[start : start + self.batch_size]))
        if not chunks:
            return {"step_logit": torch.zeros(0), "outcome_logit": torch.zeros(0)}
        return {key: torch.cat([c[key] for c in chunks]) for key in chunks[0]}

    def _combined_log_score(self, scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        step = self.step_calibrator.apply(scores["step_logit"]).clamp(1e-6, 1 - 1e-6)
        value = step.log()
        if self.outcome_weight:
            outcome = self.outcome_calibrator.apply(scores["outcome_logit"]).clamp(1e-6, 1 - 1e-6)
            value = value + self.outcome_weight * outcome.log()
        return value

    def score_states(self, states: Sequence[PolicyState]) -> torch.Tensor:
        return self._combined_log_score(self._score_texts([s.render() for s in states]))

    def score_step_completion(self, state: PolicyState) -> float:
        return float(self.score_states([state])[0])

    def score_step_candidates(
        self, question: str, history: Sequence[str], candidates: Sequence[str]
    ) -> torch.Tensor:
        if not candidates:
            return torch.zeros(0)
        states = [PolicyState(question, list(history), c) for c in candidates]
        return self.score_states(states)

    def predict_step_probability(self, state: PolicyState) -> float:
        scores = self._score_texts([state.render()])
        return float(self.step_calibrator.apply(scores["step_logit"])[0])

    def predict_outcome_probability(self, state: PolicyState) -> float:
        scores = self._score_texts([state.render()])
        return float(self.outcome_calibrator.apply(scores["outcome_logit"])[0])

    # -- token level ------------------------------------------------------
    def score_tokens(self, state: PolicyState, token_ids: Sequence[int]) -> torch.Tensor:
        ids = list(token_ids)
        if not ids:
            return torch.zeros(0)
        if self.token_scoring == "head":
            if self.model.token_head is None:
                raise RuntimeError("policy has no token head; use token_scoring='lookahead'")
            scores = self.model.score_texts([state.render()])
            return scores["token_logits"][0, torch.tensor(ids)]
        if self.base_tokenizer is None:
            raise RuntimeError("lookahead token scoring needs the base model tokenizer")
        pieces = [self.base_tokenizer.decode([token_id]) for token_id in ids]
        states = [state.with_step(state.current_step + piece) for piece in pieces]
        return self.score_states(states)

    # -- boundaries and outcome -------------------------------------------
    def action_scores(self, state: PolicyState) -> Dict[Action, float]:
        if self.model.action_head is None:
            raise RuntimeError("policy has no action head")
        scores = self.model.score_texts([state.render()])
        logits = torch.log_softmax(scores["action_logits"][0], dim=-1)
        return {action: float(logits[i]) for i, action in enumerate(ACTIONS)}

    def score_final_answer(self, trajectory: Trajectory | PolicyState) -> float:
        if isinstance(trajectory, Trajectory):
            state = PolicyState(
                question=trajectory.question,
                history=trajectory.step_texts,
                final_answer=trajectory.final_answer,
            )
        else:
            state = trajectory
        scores = self._score_texts([state.render()])
        return float(self.outcome_calibrator.apply(scores["outcome_logit"])[0])


class VerifierPolicy(PolicyAdapter):
    """Oracle policy that calls the verifier instead of predicting it.

    Not usable at scale (it pays a Lean call per candidate) but it is the
    natural skyline for the evaluation protocol: it shows what a *perfect*
    process reward model would buy.
    """

    supports_token_scores = False

    def __init__(self, verifier: Any, verified_score: float = 0.0, rejected_score: float = -5.0):
        self.verifier = verifier
        self.verified_score = verified_score
        self.rejected_score = rejected_score

    def score_step_completion(self, state: PolicyState) -> float:
        result = self.verifier.verify(state.question, state.history, state.current_step)
        return self.verified_score if result.verified else self.rejected_score

    def score_step_candidates(
        self, question: str, history: Sequence[str], candidates: Sequence[str]
    ) -> torch.Tensor:
        results = self.verifier.verify_many(question, list(history), list(candidates))
        return torch.tensor(
            [self.verified_score if r.verified else self.rejected_score for r in results]
        )

    def score_final_answer(self, trajectory: Trajectory | PolicyState) -> float:
        return 0.0

    def action_scores(self, state: PolicyState) -> Dict[Action, float]:
        return {Action.CONTINUE_STEP: 0.0, Action.CLOSE_STEP: 0.0, Action.END_ANSWER: -1.0}


class UniformPolicy(PolicyAdapter):
    """No-op policy; equivalent to alpha = 0 (plain base-model decoding)."""

    supports_token_scores = True

    def score_step_completion(self, state: PolicyState) -> float:
        return 0.0

    def score_step_candidates(
        self, question: str, history: Sequence[str], candidates: Sequence[str]
    ) -> torch.Tensor:
        return torch.zeros(len(candidates))

    def score_tokens(self, state: PolicyState, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.zeros(len(list(token_ids)))

    def score_final_answer(self, trajectory: Trajectory | PolicyState) -> float:
        return 0.0

    def action_scores(self, state: PolicyState) -> Dict[Action, float]:
        return {action: 0.0 for action in ACTIONS}


# --------------------------------------------------------------------------
# Score combination (section 8)
# --------------------------------------------------------------------------


def select_top_candidates(logits: torch.Tensor, k: int = 16, top_p: Optional[float] = None) -> torch.Tensor:
    """Candidate next tokens: top-k, optionally restricted to a nucleus."""
    k = min(k, logits.numel())
    values, indices = torch.topk(logits, k)
    if top_p is not None:
        probabilities = torch.softmax(values, dim=-1)
        keep = (probabilities.cumsum(dim=-1) - probabilities) < top_p
        keep[0] = True
        indices = indices[keep]
    return indices


def combine_scores(
    base_logits: torch.Tensor, policy_scores: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """log p'(a) = log p_base(a) + alpha * log pi_phi(a | state).

    Both sides are normalised to log-probabilities first, so `alpha` has the
    same meaning regardless of how the policy's raw scores are scaled.
    """
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    policy_log_probs = torch.log_softmax(policy_scores, dim=-1)
    return base_log_probs + alpha * policy_log_probs


def combine_reward_logits(
    base_logits: torch.Tensor, policy_rewards: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """Variant for a policy that emits arbitrary scalar rewards, not logits."""
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    return base_log_probs + alpha * policy_rewards


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_policy(
    path: str,
    model: ProcessRewardModel,
    step_calibrator: Optional[TemperatureScaler] = None,
    outcome_calibrator: Optional[TemperatureScaler] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Save the reward model together with its calibration."""
    import json
    from pathlib import Path

    directory = Path(path)
    model.save(directory)
    payload = {
        "step": (step_calibrator or TemperatureScaler()).to_dict(),
        "outcome": (outcome_calibrator or TemperatureScaler()).to_dict(),
        **(extra or {}),
    }
    (directory / "calibration.json").write_text(json.dumps(payload, indent=2))
    return str(directory)


def load_policy(
    path: str,
    outcome_weight: float = 0.5,
    token_scoring: str = "lookahead",
    base_tokenizer: Any = None,
) -> "ProcessRewardPolicy":
    """Load a saved reward model and wrap it as a decoding policy."""
    import json
    from pathlib import Path

    directory = Path(path)
    model = ProcessRewardModel.load(directory)
    calibration_path = directory / "calibration.json"
    step, outcome = TemperatureScaler(), TemperatureScaler()
    if calibration_path.exists():
        payload = json.loads(calibration_path.read_text())
        step = TemperatureScaler(**{k: v for k, v in payload.get("step", {}).items() if k in ("temperature", "bias")})
        outcome = TemperatureScaler(
            **{k: v for k, v in payload.get("outcome", {}).items() if k in ("temperature", "bias")}
        )
    return ProcessRewardPolicy(
        model,
        outcome_weight=outcome_weight,
        step_calibrator=step,
        outcome_calibrator=outcome,
        token_scoring=token_scoring,
        base_tokenizer=base_tokenizer,
    )
