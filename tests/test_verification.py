import pytest

from tests.conftest import requires_lean

from rllf.verification.base import FORMALIZATION_FAILED, LEAN_CHECK_FAILED, VERIFIED
from rllf.verification.lean_verifier import (
    CAT_FALSE_PROPOSITION,
    CAT_OK,
    LeanConfig,
    LeanStepVerifier,
    MockStepVerifier,
    VerificationCache,
    build_source,
    classify_error,
    parse_lean_messages,
)


@pytest.fixture
def mock():
    return MockStepVerifier()


def test_verified_step(mock):
    result = mock.verify("q", [], "so 3 * 4 = 12 pens")
    assert result.status == VERIFIED
    assert result.reward == 1.0
    assert result.formalization


def test_false_step_is_lean_check_failed(mock):
    result = mock.verify("q", [], "so 3 * 4 = 13 pens")
    assert result.status == LEAN_CHECK_FAILED
    assert result.reward == 0.0
    assert result.diagnostics["category"] == CAT_FALSE_PROPOSITION


def test_unformalizable_step(mock):
    result = mock.verify("q", [], "the answer looks reasonable")
    assert result.status == FORMALIZATION_FAILED
    assert result.formalization is None
    assert result.reward == 0.0


def test_conditional_step_uses_history(mock):
    assert mock.verify("q", ["We know y - 3 = 5."], "Therefore y = 8.").verified
    assert not mock.verify("q", ["We know y - 3 = 5."], "Therefore y = 9.").verified


def test_verify_many_matches_verify(mock):
    steps = ["so 2 + 2 = 4", "so 2 + 2 = 5"]
    assert [r.verified for r in mock.verify_many("q", [], steps)] == [True, False]


def test_build_source_maps_tactics_to_lines():
    source, lines = build_source("(3 : Nat) = 3", ["decide", "rfl"], header="-- header")
    body = source.splitlines()
    assert body[lines["decide"] - 1].endswith("by decide")
    assert body[lines["rfl"] - 1].endswith("by rfl")


def test_parse_lean_messages():
    messages = parse_lean_messages("f.lean:3:5: error: unsolved goals\n  x = 1\nf.lean:4:0: warning: hm")
    assert [(m.line, m.severity) for m in messages] == [(3, "error"), (4, "warning")]
    assert "unsolved goals" in messages[0].text


def test_classify_error():
    assert classify_error("Tactic `decide` proved that the proposition\n x\nis false") == CAT_FALSE_PROPOSITION
    assert classify_error("unsolved goals") == "unsolved_goal"
    assert classify_error("unknown identifier 'foo'") == "elaboration_error"


def test_cache_key_covers_question_history_and_step():
    key = VerificationCache.make_key
    assert key("q", ["a"], "s") == key("q", ["a"], "s")
    assert key("q", ["a"], "s") != key("q", ["b"], "s")
    assert key("q", ["a"], "s") != key("q2", ["a"], "s")


def test_cache_round_trips_through_disk(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = VerificationCache(path)
    cache.put("k", {"status": VERIFIED, "reward": 1.0})
    assert VerificationCache(path).get("k")["reward"] == 1.0


@requires_lean
def test_lean_verifies_true_arithmetic(tmp_path):
    verifier = LeanStepVerifier(
        config=LeanConfig(timeout=90), cache=VerificationCache(tmp_path / "c.jsonl")
    )
    result = verifier.verify("q", [], "She has 3 boxes of 4, so 3 * 4 = 12.")
    assert result.status == VERIFIED
    assert result.diagnostics["category"] == CAT_OK
    assert result.diagnostics["tactic"]


@requires_lean
def test_lean_rejects_false_arithmetic(tmp_path):
    verifier = LeanStepVerifier(
        config=LeanConfig(timeout=90), cache=VerificationCache(tmp_path / "c.jsonl")
    )
    result = verifier.verify("q", [], "She has 3 boxes of 4, so 3 * 4 = 13.")
    assert result.status == LEAN_CHECK_FAILED
    assert result.reward == 0.0


@requires_lean
def test_lean_results_are_cached(tmp_path):
    verifier = LeanStepVerifier(
        config=LeanConfig(timeout=90), cache=VerificationCache(tmp_path / "c.jsonl")
    )
    step = "so 2 + 2 = 4"
    verifier.verify("q", [], step)
    assert verifier.verify("q", [], step).diagnostics.get("cached") is True
