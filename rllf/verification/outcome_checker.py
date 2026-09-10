"""Outcome-level reward: is the final answer correct?

Deliberately independent of the step-level check.  A trajectory can be locally
verifiable and still land on the wrong number, and a correct number can be
reached through steps no formalizer can handle -- keeping the two signals apart
is what lets the policy optimise process quality and final correctness
separately.

Checkers are composed in a ladder: exact match after normalisation, numeric
equality, symbolic equivalence, and finally any user-supplied trusted solver.
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence

from ..protocol import extract_final_answer
from .base import OutcomeChecker, OutcomeVerification

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_UNIT_WORDS = re.compile(
    r"\b(dollars?|cents?|euros?|pounds?|units?|items?|apples?|books?|pens?|hours?|"
    r"minutes?|seconds?|days?|weeks?|months?|years?|miles?|kilometers?|km|meters?|m|"
    r"cm|feet|ft|inches|in|litres?|liters?|l|kg|kilograms?|grams?|g|pieces?|people|"
    r"students?|children|times?|percent)\b",
    re.IGNORECASE,
)
_NUM_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_answer(answer: Any) -> str:
    """Canonical surface form used for exact matching."""
    text = str(answer).strip()
    boxed = _BOXED_RE.findall(text)
    if boxed:
        text = boxed[-1]
    text = text.replace("$", " ").replace("%", " ").replace("\\%", " ")
    text = text.replace("\\left", "").replace("\\right", "")
    text = re.sub(r"^(the\s+)?(final\s+)?answer(\s+is)?\s*[:=]?", "", text, flags=re.IGNORECASE)
    text = _UNIT_WORDS.sub(" ", text)
    text = _NUM_COMMA_RE.sub("", text)
    text = text.strip().strip(".").strip()
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"-?\d+\.0+", text):  # 12.0 -> 12
        text = text.split(".")[0]
    return text.lower()


def to_number(text: str) -> Optional[float]:
    try:
        return float(normalize_answer(text))
    except (TypeError, ValueError):
        return None


class ExactMatchChecker(OutcomeChecker):
    """String equality after normalisation."""

    method = "exact_match"

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        answer = normalize_answer(extract_final_answer(str(final_answer)))
        expected = normalize_answer(reference)
        return OutcomeVerification(
            correct=bool(expected) and answer == expected,
            normalized_answer=answer,
            normalized_reference=expected,
            method=self.method,
        )


class NumericChecker(OutcomeChecker):
    """Numeric equality within a tolerance."""

    method = "numeric"

    def __init__(self, rel_tol: float = 1e-6, abs_tol: float = 1e-9, number_fallback: bool = True):
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        # Models answer in prose ("Anne is 8 years old."), so when the whole
        # string is not a number, fall back to the last number in it -- the
        # standard GSM8K convention.  Without this, correct answers are scored
        # wrong, which corrupts both the outcome labels and the headline metric.
        self.number_fallback = number_fallback

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        import math

        answer_text = normalize_answer(extract_final_answer(str(final_answer)))
        expected_text = normalize_answer(reference)
        answer, expected = to_number(answer_text), to_number(expected_text)
        if answer is None and expected is not None and self.number_fallback:
            numbers = _TRAILING_NUMBER_RE.findall(answer_text)
            if numbers:
                answer = float(numbers[-1])
        if answer is None or expected is None:
            return OutcomeVerification(
                correct=False,
                normalized_answer=answer_text,
                normalized_reference=expected_text,
                method=self.method,
                error="not numeric",
            )
        return OutcomeVerification(
            correct=math.isclose(answer, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol),
            normalized_answer=answer_text,
            normalized_reference=expected_text,
            method=self.method,
        )


class SymbolicChecker(OutcomeChecker):
    """Symbolic equivalence via sympy (`1/2` == `0.5` == `\\frac{1}{2}`)."""

    method = "symbolic"

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        from .formalizer import normalize_math_text

        answer_text = normalize_answer(extract_final_answer(str(final_answer)))
        expected_text = normalize_answer(reference)
        try:
            from sympy import simplify, sympify

            answer = sympify(normalize_math_text(answer_text), rational=True)
            expected = sympify(normalize_math_text(expected_text), rational=True)
            correct = bool(simplify(answer - expected) == 0)
            error = None
        except Exception as exc:
            correct, error = False, f"{type(exc).__name__}: {exc}"
        return OutcomeVerification(
            correct=correct,
            normalized_answer=answer_text,
            normalized_reference=expected_text,
            method=self.method,
            error=error,
        )


class CallableChecker(OutcomeChecker):
    """Wraps a trusted reference solver or domain-specific checker."""

    def __init__(self, fn: Callable[[str, str, Any], bool], method: str = "custom"):
        self.fn = fn
        self.method = method

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        answer = normalize_answer(extract_final_answer(str(final_answer)))
        try:
            correct, error = bool(self.fn(question, final_answer, reference)), None
        except Exception as exc:
            correct, error = False, f"{type(exc).__name__}: {exc}"
        return OutcomeVerification(
            correct=correct,
            normalized_answer=answer,
            normalized_reference=normalize_answer(reference),
            method=self.method,
            error=error,
        )


class LeanOutcomeChecker(OutcomeChecker):
    """Checks `answer = reference` with the Lean kernel.

    Useful when the reference is itself an expression (`(3+4)/2`) and you want
    the same trust anchor for the outcome reward as for the step rewards.
    """

    method = "lean"

    def __init__(self, verifier: Any):
        self.verifier = verifier

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        answer = normalize_answer(extract_final_answer(str(final_answer)))
        expected = normalize_answer(reference)
        claim = f"{answer} = {expected}"
        result = self.verifier.verify(question, [], f"Therefore {claim}.")
        return OutcomeVerification(
            correct=result.verified,
            normalized_answer=answer,
            normalized_reference=expected,
            method=self.method,
            error=result.error,
        )


class CompositeOutcomeChecker(OutcomeChecker):
    """Accepts the answer if any checker in the ladder accepts it."""

    def __init__(self, checkers: Optional[Sequence[OutcomeChecker]] = None):
        self.checkers: List[OutcomeChecker] = list(
            checkers if checkers is not None else (ExactMatchChecker(), NumericChecker(), SymbolicChecker())
        )

    def check(self, question: str, final_answer: str, reference: Any) -> OutcomeVerification:
        last: Optional[OutcomeVerification] = None
        for checker in self.checkers:
            result = checker.check(question, final_answer, reference)
            if result.correct:
                return result
            last = last or result
        return last or OutcomeVerification(correct=False, method="none", error="no checkers")


def default_outcome_checker() -> CompositeOutcomeChecker:
    return CompositeOutcomeChecker()
