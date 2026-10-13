import pytest

from rllf.generation.beam_search import (
    INTERVENE_CLASSIFIER,
    INTERVENE_HYBRID,
    INTERVENE_VERIFIER,
    BeamSearchConfig,
    lean_constrained_beam_search,
)
from rllf.verification.lean_verifier import MockStepVerifier
from rllf.verification.outcome_checker import default_outcome_checker

GOOD = "so 2 * 3 = 6"
BAD = "so 2 * 3 = 7"


class _ScriptedBeamModel:
    """`generate_step_beam` returns fixed `(text, lm_score)` pairs keyed by
    `(question, history)` -- no real generation, mirrors `ScriptedBaseModel`'s
    role for the `<step>`-candidate decoder in `test_decoder.py`."""

    def __init__(self, script):
        self.script = script  # {(question, history_tuple): [(text, lm_score), ...]}

    def generate_step_beam(self, question, history, num_beams=4, num_return_sequences=4, **kwargs):
        candidates = self.script.get((question, tuple(history)), [])
        return list(candidates[:num_return_sequences])


class _KeywordBeamPolicy:
    """`predict_step_logit_for_beam` favours any step containing `target`."""

    def __init__(self, target, magnitude=5.0):
        self.target = target
        self.magnitude = magnitude

    def predict_step_logit_for_beam(self, state):
        return self.magnitude if self.target in state.current_step else -self.magnitude

    def predict_step_probability(self, state):
        return 1.0 if self.target in state.current_step else 0.0


class _ThresholdPolicy:
    """Confident (>=0.5) everywhere except on a step containing `unsure_marker`."""

    def __init__(self, unsure_marker):
        self.unsure_marker = unsure_marker

    def predict_step_probability(self, state):
        return 0.0 if self.unsure_marker in state.current_step else 1.0

    def predict_step_logit_for_beam(self, state):
        return 1.0


def test_classifier_intervention_reranks_and_prunes_across_a_boundary():
    model = _ScriptedBeamModel({
        ("q", ()): [("first step", -1.0)],
        ("q", ("first step",)): [("weak second step", -1.0), ("GOOD second step", -3.0)],
    })
    # The base LM prefers "weak second step" (higher/less-negative log-prob);
    # the classifier strongly prefers "GOOD second step" -- with alpha=1 and
    # a large enough classifier margin, the classifier's pick should win.
    policy = _KeywordBeamPolicy(target="GOOD")
    config = BeamSearchConfig(beam_width=1, expansion_per_beam=2, alpha=1.0, max_steps=2)
    result = lean_constrained_beam_search("q", None, model, policy, config=config)
    assert result.trajectory.step_texts == ["first step", "GOOD second step"]


def test_classifier_never_fires_on_a_final_answer():
    model = _ScriptedBeamModel({("q", ()): [(r"work \boxed{6}", -1.0)]})
    # A classifier that would reject every ordinary step if it were consulted.
    policy = _KeywordBeamPolicy(target="NEVER_MATCHES_ANYTHING")
    config = BeamSearchConfig(beam_width=1, max_steps=3)
    result = lean_constrained_beam_search(
        "q", "6", model, policy, outcome_checker=default_outcome_checker(), config=config
    )
    assert result.trajectory.step_texts == []  # the final answer is not a reasoning step
    assert result.trajectory.final_answer == r"work \boxed{6}"
    assert result.trajectory.outcome_reward == 1.0
    assert result.trajectory.metadata["final_classifier_score"] == 0.0


def test_verifier_intervention_rejects_a_failing_step():
    model = _ScriptedBeamModel({
        ("q", ()): [("start", -1.0)],
        ("q", ("start",)): [(BAD, -1.0), (GOOD, -3.0)],  # LM prefers BAD (less negative)
    })
    config = BeamSearchConfig(
        beam_width=1, expansion_per_beam=2, intervention=INTERVENE_VERIFIER, max_steps=2,
    )
    result = lean_constrained_beam_search(
        "q", None, model, policy=None, verifier=MockStepVerifier(), config=config,
    )
    assert result.trajectory.step_texts == ["start", GOOD]


def test_verifier_intervention_needs_a_verifier():
    model = _ScriptedBeamModel({("q", ()): [(GOOD, -1.0)]})
    config = BeamSearchConfig(beam_width=1, intervention=INTERVENE_VERIFIER, max_steps=1)
    with pytest.raises(ValueError):
        lean_constrained_beam_search("q", None, model, policy=None, config=config)


def test_hybrid_intervention_escalates_only_the_unsure_beam_to_the_verifier():
    model = _ScriptedBeamModel({
        ("q", ()): [("start", -1.0)],
        ("q", ("start",)): [(GOOD, -1.0), (f"{BAD} UNSURE", -1.0)],
    })
    policy = _ThresholdPolicy(unsure_marker="UNSURE")
    config = BeamSearchConfig(
        beam_width=2, expansion_per_beam=2,
        intervention=INTERVENE_HYBRID, hybrid_threshold=0.5, max_steps=2,
    )
    result = lean_constrained_beam_search(
        "q", None, model, policy, verifier=MockStepVerifier(), config=config,
    )
    # Only the one candidate below the confidence threshold should have paid
    # for a real Lean call -- the confident one is scored by the classifier alone.
    assert result.stats.lean_calls == 1


def test_max_steps_terminates_the_loop():
    model = _ScriptedBeamModel({
        ("q", ()): [("s1", -1.0)],
        ("q", ("s1",)): [("s2", -1.0)],
    })
    policy = _KeywordBeamPolicy(target="never present")
    config = BeamSearchConfig(beam_width=1, max_steps=2)
    result = lean_constrained_beam_search("q", None, model, policy, config=config)
    assert result.trajectory.step_texts == ["s1", "s2"]
    assert result.stats.steps_attempted == 2


def test_max_steps_none_is_bounded_only_by_the_safety_ceiling(monkeypatch):
    import rllf.generation.beam_search as beam_search_module

    monkeypatch.setattr(beam_search_module, "MAX_STEPS_SAFETY_CEILING", 3)
    model = _ScriptedBeamModel({
        ("q", ()): [("s1", -1.0)],
        ("q", ("s1",)): [("s2", -1.0)],
        ("q", ("s1", "s2")): [("s3", -1.0)],
        ("q", ("s1", "s2", "s3")): [("s4", -1.0)],  # never reached
    })
    policy = _KeywordBeamPolicy(target="never present")
    config = BeamSearchConfig(beam_width=1, max_steps=None)
    result = lean_constrained_beam_search("q", None, model, policy, config=config)
    assert result.trajectory.step_texts == ["s1", "s2", "s3"]
    assert result.stats.steps_attempted == 3


def test_all_beams_finishing_early_stops_before_max_steps():
    model = _ScriptedBeamModel({("q", ()): [(r"\boxed{1}", -1.0)]})
    policy = _KeywordBeamPolicy(target="never present")
    config = BeamSearchConfig(beam_width=1, max_steps=8)
    result = lean_constrained_beam_search("q", None, model, policy, config=config)
    assert result.stats.steps_attempted == 1


def test_stats_record_generated_tokens_and_are_attached_to_the_trajectory():
    model = _ScriptedBeamModel({("q", ()): [("a short step", -1.0)]})
    policy = _KeywordBeamPolicy(target="never present")
    config = BeamSearchConfig(beam_width=1, max_steps=1)
    result = lean_constrained_beam_search("q", None, model, policy, config=config)
    assert result.stats.generated_tokens == len("a short step".split())
    assert result.trajectory.metadata["generated_tokens"] == result.stats.generated_tokens
    assert result.trajectory.metadata["mode"] == "lean_constrained_beam_search"


class _ConstantBeamPolicy:
    def __init__(self, value):
        self.value = value

    def predict_step_logit_for_beam(self, state):
        return self.value


class _NeverEndingModel:
    """Every prefix offers a final answer (better LM score) and another step."""

    def generate_step_beam(self, question, history, num_beams=4, num_return_sequences=4, **kwargs):
        return [(r"\boxed{6}", -0.1), (f"step {len(history)}", -0.2)][:num_return_sequences]


@pytest.mark.parametrize("aggregate, expect_steps", [("sum", 20), ("mean", 1)])
def test_classifier_aggregate_controls_the_length_preference(aggregate, expect_steps):
    # A positive per-step classifier score summed over steps lets an unfinished
    # beam outscore every finished one, so the search never chooses to stop;
    # averaging removes that preference and the better-LM final answer wins.
    config = BeamSearchConfig(beam_width=2, expansion_per_beam=2, alpha=1.0, max_steps=20,
                              classifier_aggregate=aggregate)
    result = lean_constrained_beam_search("q", None, _NeverEndingModel(), _ConstantBeamPolicy(1.0), config=config)
    assert len(result.trajectory.step_texts) == expect_steps


def test_unknown_classifier_aggregate_is_rejected():
    with pytest.raises(ValueError):
        lean_constrained_beam_search("q", None, _NeverEndingModel(), _ConstantBeamPolicy(1.0),
                                     config=BeamSearchConfig(classifier_aggregate="max"))
