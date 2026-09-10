import pytest

from rllf.verification.formalizer import (
    NO_CLAIM,
    UNSUPPORTED,
    LLMFormalizer,
    RuleBasedFormalizer,
    choose_numeric_type,
    extract_relations,
    normalize_math_text,
)


@pytest.fixture
def formalizer():
    return RuleBasedFormalizer()


def test_extracts_ground_arithmetic_claim(formalizer):
    result = formalizer.formalize("q", [], "She buys 3 packs of 4 pens, so 3 * 4 = 12 pens.")
    assert result.ok
    assert result.statement == "(3 * 4 : Nat) = 12"


def test_does_not_simplify_the_claim(formalizer):
    """Simplifying before Lean would turn the check into a tautology."""
    result = formalizer.formalize("q", [], "so 3 * 4 = 12")
    assert "3 * 4" in result.statement


def test_prose_equation_is_not_a_math_claim(formalizer):
    result = formalizer.formalize("q", [], "Then the total cost = 12 dollars.")
    assert not result.ok
    assert result.error == NO_CLAIM


def test_conditional_claim_becomes_an_implication(formalizer):
    result = formalizer.formalize("q", [], "Since x + 2 = 5, we get x = 3.")
    assert result.statement == "∀ (x : Nat), x + 2 = 5 → x = 3"
    assert result.tactic_hints[0] == "omega"


def test_history_supplies_the_defining_equation(formalizer):
    result = formalizer.formalize("q", ["We know y - 3 = 5."], "Therefore y = 8.")
    assert result.ok
    assert "y - 3 = 5" in result.statement


def test_unbound_variable_is_reported_not_fabricated(formalizer):
    result = formalizer.formalize("q", [], "Then z = 4.")
    assert not result.ok
    assert result.error == UNSUPPORTED
    assert result.diagnostics["unbound_variables"] == ["z"]


def test_latex_is_normalized(formalizer):
    result = formalizer.formalize("q", [], r"So $\frac{12}{4} = 3$.")
    assert result.ok
    assert result.diagnostics["numeric_type"] == "Rat"


def test_numeric_type_avoids_nat_truncation():
    assert choose_numeric_type(["3 + 4 = 7"]) == "Nat"
    assert choose_numeric_type(["5 - 8 = -3"]) == "Int"
    assert choose_numeric_type(["10 / 4 = 2.5"]) == "Rat"


def test_inequalities_are_supported(formalizer):
    result = formalizer.formalize("q", [], "Since 100 > 99 the claim holds.")
    assert result.statement == "(100 : Nat) > 99"


def test_chained_relations_yield_multiple_claims():
    relations = extract_relations("We have 3 * 4 = 12 and 12 + 1 = 13.")
    assert [r.raw for r in relations] == ["3 * 4 = 12", "12 + 1 = 13"]


def test_normalize_math_text_handles_unicode_and_commas():
    assert "1000 * 2" in normalize_math_text("1,000 × 2")


class _EchoGenerator:
    def __init__(self, text):
        self.text = text

    def generate(self, prompt, **kwargs):
        return self.text


def test_llm_formalizer_cleans_the_generation():
    formalizer = LLMFormalizer(_EchoGenerator("```lean\ntheorem foo : (2:Nat) + 2 = 4 := by decide\n```"))
    result = formalizer.formalize("q", [], "two plus two is four")
    assert result.ok
    assert result.statement == "(2:Nat) + 2 = 4"


def test_llm_formalizer_falls_back():
    class _Broken:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("no model")

    formalizer = LLMFormalizer(_Broken(), fallback=RuleBasedFormalizer())
    result = formalizer.formalize("q", [], "so 3 * 4 = 12")
    assert result.ok
    assert "llm_error" in result.diagnostics


def test_implicit_multiplication_becomes_explicit(formalizer):
    """`4B` and `4(3)` are products in prose but syntax errors in Lean."""
    result = formalizer.formalize("q", [], r"Now \(B = 5\) so \[R = 4B\] gives \[R = 20\]")
    assert result.ok
    assert "4 * B" in result.statement
    assert "4B" not in result.statement


def test_implicit_multiplication_across_parentheses(formalizer):
    result = formalizer.formalize("q", [], r"Substituting, \[ S = 4(3) - 5 = 7 \]")
    assert "4 * (3)" in result.statement


def test_function_application_is_not_read_as_multiplication():
    """`f(x)` is application, not a product.

    Rewriting it to `f * (x)` yields Lean that elaborates cleanly while meaning
    something else -- the one failure a verifier-based reward cannot catch.
    """
    from rllf.verification.formalizer import normalize_math_text

    assert normalize_math_text("f(2x+3) = 2f(x) + 3") == "f(2 * x+3) = 2 * f(x) + 3"
    # A numeral or a closing paren on the left is unambiguous multiplication.
    assert "4 * (3)" in normalize_math_text("4(3)")
    assert "(a) * (b)" in normalize_math_text("(a)(b)")


def test_tactic_failure_is_not_an_elaboration_error():
    from rllf.verification.lean_verifier import CAT_UNSOLVED_GOAL, classify_error

    assert classify_error("omega could not prove the goal: a counterexample") == CAT_UNSOLVED_GOAL
    assert classify_error("linarith failed to prove the goal") == CAT_UNSOLVED_GOAL


def test_juxtaposition_rule_does_not_fire_across_prose():
    from rllf.verification.formalizer import normalize_math_text

    # A space means two words, not a product; inventing `*` here would
    # manufacture claims out of English.
    assert "5 * pens" not in normalize_math_text("cost 5 pens")
    assert "(3) * (4)" in normalize_math_text("(3)(4)")


def test_formalizer_version_is_in_the_cache_key():
    from rllf.verification.formalizer import FORMALIZER_VERSION
    from rllf.verification.lean_verifier import LeanStepVerifier

    verifier = LeanStepVerifier()
    assert str(FORMALIZER_VERSION) in verifier._cache_salt()
