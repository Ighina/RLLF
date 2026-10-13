import pytest

from rllf.verification.math_equivalence import latex_to_sympy_text, math_equivalent
from rllf.verification.outcome_checker import MathEquivalenceChecker, default_outcome_checker


@pytest.mark.parametrize(
    "answer, reference, expected",
    [
        ("cot(-60°) = -(√3)/3.", r"-\frac{\sqrt{3}}{3}", True),
        ("-√3/3", r"-\frac{\sqrt{3}}{3}", True),
        ("-1/√3", r"-\frac{\sqrt{3}}{3}", True),
        (r"-\frac{\sqrt2}{2}", r"-\frac{1}{\sqrt{2}}", True),
        ("2π/5", r"\frac{2\pi}{5}", True),
        ("(C)", r"\text{(C)}", True),
        ("7.5°", r"7.5^\circ", True),
        ("2, 4", "2,4", True),
        ("4,2", "2,4", False),
        ("0.5", r"\frac12", True),
        ("1/2", r"\dfrac{1}{2}", True),
        ("x^2 + 2x", "x^2+2x", True),
        ("x^2 + 2x", "x^2+3x", False),
        (r"\$3.50", "3.50", True),
        ("3", "4", False),
        (r"\begin{pmatrix} -5 \\ 9 \\ -8 \end{pmatrix}", r"\begin{pmatrix} -5 \\ 9 \\ -8 \end{pmatrix}", True),
        ("[0, 1)", "[0,1)", True),
        ("1,000", "1000", True),
        (r"\frac{3}{\sqrt{10}}", r"\frac{3\sqrt{10}}{10}", True),
        ("12", r"12 \text{ units}", True),
        (r"6\sqrt{2}", r"6\sqrt2", True),
        ("x=5", "5", True),
        ("", "3", None),
        ("Final answer: -√3/3.", r"-\frac{\sqrt{3}}{3}", True),
        ("Therefore, the period of y = sin(5x) is 2π/5.", r"\frac{2\pi}{5}", True),
        ("The correct answer is (B).", r"\text{(B)}", True),
        ("Therefore, the period of y=sin x + cos x is 2π.", r"2 \pi", True),
        (r"Final answer: \(\begin{pmatrix}-3\\4\end{pmatrix}.\)", r"\begin{pmatrix} -3 \\ 4 \end{pmatrix}", True),
        ("|(12-9i)(8+15i)| = 249.", "255", False),
        ("Therefore, sec(5π/3)=1.", "2", False),
    ],
)
def test_math_equivalent(answer, reference, expected):
    assert math_equivalent(answer, reference) is expected


def test_latex_to_sympy_text_nested_fraction():
    assert math_equivalent(r"\frac{\sqrt{3}}{\frac{1}{2}}", r"2\sqrt{3}") is True
    assert "sqrt(3)" in latex_to_sympy_text(r"\frac{\sqrt{3}}{\frac{1}{2}}")


def test_default_checker_accepts_latex_equivalents_via_last_rung():
    checker = default_outcome_checker()
    result = checker.check("q", "<final> Final answer: cot(-60°) = -(√3)/3.", r"-\frac{\sqrt{3}}{3}")
    assert result.correct and result.method == "math_equivalence"
    # earlier rungs still win when they can
    assert checker.check("q", "<final> 42", "42").method == "exact_match"
    assert not checker.check("q", "<final> 41", "42").correct
    assert MathEquivalenceChecker().check("q", "", "3").error == "unparseable"
