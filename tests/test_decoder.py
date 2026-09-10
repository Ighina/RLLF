import torch

from rllf.generation.base_model import ScriptedBaseModel, ToyCharModel, truncate_at_boundary
from rllf.generation.guided_decoder import (
    MODE_PLAIN,
    MODE_REJECTION,
    MODE_RERANK,
    MODE_TOKEN,
    VERIFY_DEFERRED,
    VERIFY_HYBRID,
    VERIFY_NONE,
    VERIFY_ONLINE,
    DecodeConfig,
    GuidedDecoder,
    best_of_n,
    sample_verified_step,
)
from rllf.policy.scoring import PolicyAdapter, UniformPolicy
from rllf.protocol import STEP_TOKEN
from rllf.verification.lean_verifier import MockStepVerifier
from rllf.verification.outcome_checker import default_outcome_checker

GOOD = "so 2 * 3 = 6"
BAD = "so 2 * 3 = 7"


class _KeywordPolicy(PolicyAdapter):
    """Scores any text containing `target` above everything else."""

    supports_token_scores = True

    def __init__(self, target):
        self.target = target

    def score_step_completion(self, state):
        return 1.0 if self.target in state.current_step else -1.0

    def score_tokens(self, state, token_ids):
        return torch.zeros(len(list(token_ids)))

    def score_final_answer(self, trajectory):
        return 0.0


def _script(steps):
    return ScriptedBaseModel({"q": steps})


def test_rerank_selects_the_policy_preferred_candidate():
    decoder = GuidedDecoder(
        _script([[BAD, GOOD], ["Final answer: 6"]]),
        policy=_KeywordPolicy(GOOD),
        config=DecodeConfig(mode=MODE_RERANK, num_candidates=2, verification=VERIFY_NONE, max_steps=3),
    )
    result = decoder.generate("q")
    assert result.steps == [GOOD]
    assert result.final_answer == "Final answer: 6"


def test_alpha_zero_reproduces_the_base_model():
    decoder = GuidedDecoder(
        _script([[BAD, GOOD], ["Final answer: 7"]]),
        policy=_KeywordPolicy(GOOD),
        config=DecodeConfig(mode=MODE_RERANK, num_candidates=2, alpha=0.0, verification=VERIFY_NONE, max_steps=3),
    )
    assert decoder.generate("q").steps == [BAD]


def test_rejection_sampling_prefers_a_verified_candidate():
    decoder = GuidedDecoder(
        _script([[BAD, GOOD], ["Final answer: 6"]]),
        policy=_KeywordPolicy(BAD),  # policy points the wrong way on purpose
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_REJECTION, num_candidates=2, verify_top_m=2, verification=VERIFY_NONE, max_steps=3),
    )
    result = decoder.generate("q")
    assert result.steps == [GOOD]
    assert result.trajectory.steps[0].verified
    assert result.stats.lean_calls > 0


def test_rejection_records_failure_when_nothing_verifies():
    decoder = GuidedDecoder(
        _script([[BAD], ["Final answer: 7"]]),
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_REJECTION, num_candidates=1, max_repairs=1, verification=VERIFY_NONE, max_steps=3),
    )
    result = decoder.generate("q")
    assert result.steps == [BAD]
    assert not result.trajectory.steps[0].verified
    assert result.stats.resamples >= 1


def test_deferred_verification_labels_every_step():
    decoder = GuidedDecoder(
        _script([[GOOD], [BAD], ["Final answer: 7"]]),
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_DEFERRED, max_steps=4),
    )
    result = decoder.generate("q")
    assert [s.verified for s in result.trajectory.steps] == [True, False]
    assert result.stats.lean_calls == 2


def test_online_verification_runs_during_decoding():
    decoder = GuidedDecoder(
        _script([[GOOD], ["Final answer: 6"]]),
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_ONLINE, max_steps=3),
    )
    result = decoder.generate("q")
    assert result.trajectory.steps[0].verified
    assert result.stats.lean_calls == 1


def test_hybrid_verification_skips_confident_steps():
    class _Confident(UniformPolicy):
        def predict_step_probability(self, state):
            return 0.99

    decoder = GuidedDecoder(
        _script([[GOOD], ["Final answer: 6"]]),
        policy=_Confident(),
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_HYBRID, hybrid_threshold=0.5, max_steps=3),
    )
    result = decoder.generate("q")
    # Skipped online, then picked up by the deferred sweep: one call, not two.
    assert result.stats.lean_calls == 1


def test_outcome_reward_is_checked_against_the_reference():
    decoder = GuidedDecoder(
        _script([[GOOD], ["Final answer: 6"]]),
        outcome_checker=default_outcome_checker(),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_NONE, max_steps=3),
    )
    assert decoder.generate("q", reference="6").outcome == 1.0
    assert decoder.generate("q", reference="7").outcome == 0.0


def test_max_steps_terminates_the_loop():
    decoder = GuidedDecoder(
        ScriptedBaseModel({"q": [[GOOD]] * 10}),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_NONE, max_steps=3),
    )
    assert len(decoder.generate("q").steps) <= 3


def test_stats_are_recorded_on_the_trajectory():
    decoder = GuidedDecoder(
        _script([[GOOD], ["Final answer: 6"]]),
        verifier=MockStepVerifier(),
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_DEFERRED, max_steps=3),
    )
    metadata = decoder.generate("q").trajectory.metadata
    assert metadata["lean_calls"] == 1.0
    assert metadata["latency_s"] >= 0.0
    assert "resample_rate" in metadata


def test_token_guidance_steers_away_from_the_base_preference():
    """The base model prefers the wrong branch; the policy must overrule it."""
    target = "2 * 3 = 6"
    distractor = "2 * 3 = 7"
    model = ToyCharModel(
        {"q": [target + STEP_TOKEN, distractor + STEP_TOKEN]},
        prior={distractor + STEP_TOKEN: 6.0},
    )

    class _CharPolicy(PolicyAdapter):
        supports_token_scores = True

        def score_tokens(self, state, token_ids):
            prefix = state.current_step
            scores = []
            for token_id in token_ids:
                piece = model.tokenizer.decode([token_id])
                scores.append(4.0 if (prefix + piece) in target + STEP_TOKEN else 0.0)
            return torch.tensor(scores)

        def score_step_completion(self, state):
            return 0.0

        def score_final_answer(self, trajectory):
            return 0.0

    unguided = GuidedDecoder(
        model, config=DecodeConfig(mode=MODE_TOKEN, alpha=0.0, max_steps=1, max_tokens_per_step=32)
    ).generate("q")
    guided = GuidedDecoder(
        model, policy=_CharPolicy(), config=DecodeConfig(mode=MODE_TOKEN, alpha=4.0, max_steps=1, max_tokens_per_step=32)
    ).generate("q")
    assert unguided.steps == [distractor]
    assert guided.steps == [target]


def test_token_mode_respects_the_minimum_step_length():
    model = ToyCharModel({"q": ["ab" + STEP_TOKEN]})
    decoder = GuidedDecoder(
        model,
        config=DecodeConfig(mode=MODE_TOKEN, alpha=0.0, max_steps=1, min_step_tokens=2, max_tokens_per_step=16),
    )
    assert decoder.generate("q").steps == ["ab"]


def test_sample_verified_step_returns_the_verified_candidate():
    step, result = sample_verified_step("q", [], _script([[BAD, GOOD]]), MockStepVerifier(), attempts=2)
    assert step == GOOD
    assert result.verified


def test_sample_verified_step_falls_back_to_the_policy_choice():
    step, result = sample_verified_step(
        "q", [], _script([[BAD, "so 2 * 3 = 8"]]), MockStepVerifier(), attempts=2, policy=_KeywordPolicy(BAD)
    )
    assert step == BAD
    assert not result.verified


def test_best_of_n_ranks_by_verification():
    result = best_of_n(
        "q",
        "6",
        ScriptedBaseModel({"q": [[GOOD, BAD], ["Final answer: 6"]]}, seed=1),
        scorer=None,
        n=4,
        verifier=MockStepVerifier(),
        outcome_checker=default_outcome_checker(),
        rank_by="verification",
        config=DecodeConfig(mode=MODE_PLAIN, verification=VERIFY_DEFERRED, max_steps=3),
    )
    assert result.trajectory.steps[0].verified
    assert result.trajectory.metadata["best_of_n"] == 4


def test_truncate_at_boundary():
    assert truncate_at_boundary(f"  a step {STEP_TOKEN} trailing") == "a step"


def test_truncate_normalises_the_boundaries_a_real_model_emits():
    """An un-finetuned model falls back to blank lines and `Step N:` headings."""
    assert truncate_at_boundary("first part.\n\nStep 2: more") == "first part."
    assert truncate_at_boundary("Step 1: find x.\nx = 4") == "find x.\nx = 4"
    assert truncate_at_boundary("do it.\nStep 2: then this") == "do it."


def test_resolve_device_returns_something_torch_accepts():
    import torch

    from rllf.generation.base_model import resolve_device

    assert torch.device(resolve_device()) is not None
    assert resolve_device("cpu") == "cpu"
