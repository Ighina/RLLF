"""Pure-Python parts of the real-Lean pipeline (no vLLM / Lean needed)."""
from rllf.verification.real_lean_pipeline import _THEOREM_RE, _splice_proof, render_step_claim

BINDER_FREE = "import Mathlib\n\ntheorem step_0 : (7 : ℝ) * 6 = 42 := by sorry"
WITH_BINDERS = (
    "import Mathlib\n\ntheorem step_0 (A B C : ℝ) \n  (h_triangle : A + B + C = Real.pi)\n"
    "  (h_positive : 0 < A ∧ 0 < B ∧ 0 < C) :\n  Real.sin C = Real.sin (A + B) := by sorry"
)
WITH_LET = "theorem step_3 (x : ℝ) (hx : x = 2) : let y : ℝ := x + 1; y = 3 := by\n  simp"


def test_theorem_regex_accepts_binders_and_lets():
    for statement in (BINDER_FREE, WITH_BINDERS, WITH_LET):
        match = _THEOREM_RE.search(statement + " :=")
        assert match, statement
        assert match.group(1).startswith("step_")
        assert statement[: match.end()].rstrip().endswith(":= by")
    # the `let ... :=` inside the statement is not mistaken for the proof boundary
    m = _THEOREM_RE.search(WITH_LET)
    assert "let y" in m.group(2)


def test_splice_proof_keeps_our_signature():
    spliced = _splice_proof(WITH_BINDERS, "\n  rw [show C = Real.pi - (A + B) by linarith]\n  simp")
    assert spliced is not None
    assert spliced.startswith(WITH_BINDERS[: WITH_BINDERS.index(":= by") + 5])
    assert "sorry" not in spliced and spliced.endswith("simp")
    assert _splice_proof("no theorem here", "simp") is None


def test_render_step_claim_default_has_no_problem_text():
    claim = render_step_claim("Find x.", ["s1", "s2"], "s3")
    assert "Find x." not in claim and "Problem" not in claim
    assert "(1) s1" in claim and "(2) s2" in claim and claim.endswith("Statement to formalize: s3")
    assert render_step_claim("Find x.", [], "s0") == "Statement to formalize: s0"


def test_render_step_claim_legacy_and_unknown_style():
    import pytest

    legacy = render_step_claim("Q?", ["s1"], "s2", style="legacy")
    assert legacy.startswith("Problem: Q?") and "0. s1" in legacy and legacy.endswith("Step to verify: s2")
    with pytest.raises(ValueError):
        render_step_claim("Q?", [], "s", style="nope")


def test_cap_heartbeats():
    from rllf.verification.real_lean_pipeline import DEFAULT_MAX_HEARTBEATS, cap_heartbeats

    code = "import Mathlib\n\nset_option maxHeartbeats 0\n\ntheorem t : 1 = 1 := by rfl"
    assert f"set_option maxHeartbeats {DEFAULT_MAX_HEARTBEATS}" in cap_heartbeats(code)
    assert "maxHeartbeats 0" not in cap_heartbeats(code)
    assert cap_heartbeats(code, None) == code and cap_heartbeats(code, 0) == code
    assert cap_heartbeats("set_option maxHeartbeats 200000\n", 5) == "set_option maxHeartbeats 200000\n"
