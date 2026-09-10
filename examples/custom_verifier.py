"""Plugging a domain-specific checker into the reward pipeline.

Every verifier is an adapter, so a project can swap the Lean step verifier or
the outcome checker without touching the policy, the decoder or the dataset.
"""

from rllf.verification.base import VERIFIED, StepVerification, StepVerifier
from rllf.verification.outcome_checker import CallableChecker, CompositeOutcomeChecker


class ParityStepVerifier(StepVerifier):
    """Toy domain rule: a step is valid when it mentions an even number."""

    def verify(self, question, previous_steps, current_step):
        import re

        numbers = [int(n) for n in re.findall(r"\d+", current_step)]
        ok = bool(numbers) and all(n % 2 == 0 for n in numbers)
        return StepVerification(
            status=VERIFIED if ok else "lean_check_failed",
            reward=1.0 if ok else 0.0,
            formalization=f"all_even {numbers}",
            error=None if ok else "found an odd number",
            diagnostics={"numbers": numbers},
        )


def reference_solver(question: str, answer: str, reference) -> bool:
    """Stand-in for a trusted solver or a CAS."""
    return str(reference) in answer


outcome_checker = CompositeOutcomeChecker([CallableChecker(reference_solver, method="solver")])

if __name__ == "__main__":
    verifier = ParityStepVerifier()
    print(verifier.verify("q", [], "we get 4 + 2 = 6").to_dict())
    print(verifier.verify("q", [], "we get 3 + 2 = 5").to_dict())
    print(outcome_checker.check("q", "Final answer: 37", "37").to_dict())
