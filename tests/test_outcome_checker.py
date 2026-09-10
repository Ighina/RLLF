import pytest

from rllf.verification.outcome_checker import (
    CallableChecker,
    CompositeOutcomeChecker,
    ExactMatchChecker,
    NumericChecker,
    SymbolicChecker,
    normalize_answer,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Final answer: 12 apples", "12"),
        ("$1,200", "1200"),
        (r"\boxed{42}", "42"),
        ("12.0", "12"),
        ("The answer is 7", "7"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_exact_match_ignores_units_and_formatting():
    result = ExactMatchChecker().check("q", "Final answer: 12 pens", "12")
    assert result.correct
    assert result.normalized_answer == "12"


def test_numeric_checker_handles_decimals():
    assert NumericChecker().check("q", "Final answer: 12.000", "12").correct
    assert not NumericChecker().check("q", "Final answer: 13", "12").correct


def test_symbolic_checker_matches_equivalent_forms():
    assert SymbolicChecker().check("q", "Final answer: 0.5", "1/2").correct
    assert SymbolicChecker().check("q", "Final answer: x+x", "2*x").correct


def test_composite_reports_the_method_that_accepted():
    checker = CompositeOutcomeChecker()
    assert checker.check("q", "Final answer: 12", "12").method == "exact_match"
    assert checker.check("q", "Final answer: 0.5", "1/2").method == "symbolic"


def test_composite_rejects_wrong_answers():
    result = CompositeOutcomeChecker().check("q", "Final answer: 13", "12")
    assert not result.correct
    assert result.reward == 0.0


def test_callable_checker_wraps_a_trusted_solver():
    checker = CallableChecker(lambda q, a, ref: "12" in a, method="solver")
    assert checker.check("q", "Final answer: 12", None).correct
    assert checker.check("q", "Final answer: 9", None).method == "solver"


def test_callable_checker_survives_a_raising_solver():
    def boom(q, a, ref):
        raise ValueError("nope")

    result = CallableChecker(boom).check("q", "1", None)
    assert not result.correct
    assert "ValueError" in result.error


def test_prose_answers_are_scored_by_their_number():
    """Models answer in prose; GSM8K convention is the last number."""
    checker = CompositeOutcomeChecker()
    assert checker.check("q", "Final answer: Anne is 8 years old.", "8").correct
    assert checker.check("q", "Final answer: she sold 72 clips altogether.", "72").correct
    assert not checker.check("q", "Final answer: Anne is 9 years old.", "8").correct


def test_number_fallback_can_be_disabled():
    strict = NumericChecker(number_fallback=False)
    assert not strict.check("q", "Final answer: Anne is 8 years old.", "8").correct
