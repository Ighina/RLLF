"""Verifier interfaces and structured results.

The methodology keeps the step-level ("is this step auto-formalizable and
locally valid?") and outcome-level ("is the final answer right?") signals
separate, because a trajectory can be locally verifiable yet wrong, or right
yet hard to formalize.  Both are expressed here so that a mock verifier and a
real Lean verifier are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Step verification statuses.
VERIFIED = "verified"
FORMALIZATION_FAILED = "formalization_failed"
LEAN_CHECK_FAILED = "lean_check_failed"
SKIPPED = "skipped"


@dataclass
class StepVerification:
    """Result of verifying a single reasoning step."""

    status: str
    reward: float
    formalization: Optional[str] = None
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "verified": self.verified,
            "reward": self.reward,
            "formalization": self.formalization,
            "error": self.error,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class OutcomeVerification:
    """Result of checking a final answer against a trusted reference."""

    correct: bool
    normalized_answer: Optional[str] = None
    normalized_reference: Optional[str] = None
    method: str = "none"
    error: Optional[str] = None

    @property
    def reward(self) -> float:
        return 1.0 if self.correct else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "reward": self.reward,
            "normalized_answer": self.normalized_answer,
            "normalized_reference": self.normalized_reference,
            "method": self.method,
            "error": self.error,
        }


class StepVerifier:
    """Interface for the step-level (auto-formalization) reward."""

    def verify(
        self, question: str, previous_steps: Sequence[str], current_step: str
    ) -> StepVerification:
        raise NotImplementedError

    def verify_many(
        self, question: str, previous_steps: Sequence[str], steps: Sequence[str]
    ) -> List[StepVerification]:
        """Verify candidate steps that all share the same history."""
        return [self.verify(question, previous_steps, s) for s in steps]

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        pass


class OutcomeChecker:
    """Interface for the outcome-level reward."""

    def check(
        self, question: str, final_answer: str, reference: Any
    ) -> OutcomeVerification:
        raise NotImplementedError


class VerifierAdapter:
    """Bundles the two signals behind the interface used by the decoder."""

    def __init__(self, step_verifier: StepVerifier, outcome_checker: OutcomeChecker):
        self.step_verifier = step_verifier
        self.outcome_checker = outcome_checker

    def verify_step(
        self, question: str, history: Sequence[str], step: str
    ) -> StepVerification:
        return self.step_verifier.verify(question, history, step)

    def verify_steps(
        self, question: str, history: Sequence[str], steps: Sequence[str]
    ) -> List[StepVerification]:
        return self.step_verifier.verify_many(question, history, steps)

    def verify_outcome(
        self, question: str, answer: str, reference: Any
    ) -> OutcomeVerification:
        return self.outcome_checker.check(question, answer, reference)


class NullStepVerifier(StepVerifier):
    """Verifier that abstains -- used to run the pipeline without Lean."""

    def verify(
        self, question: str, previous_steps: Sequence[str], current_step: str
    ) -> StepVerification:
        return StepVerification(status=SKIPPED, reward=0.0)
