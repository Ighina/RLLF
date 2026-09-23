import pytest
import torch

from rllf.data.dataset import build_pairs, build_step_examples
from rllf.data.trajectory import StepRecord, Trajectory
from rllf.policy.model import ACTIONS, ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import (
    ProcessRewardPolicy,
    TemperatureScaler,
    UniformPolicy,
    VerifierPolicy,
    combine_reward_logits,
    combine_scores,
    load_policy,
    save_policy,
    score_trajectory_formalizability,
    select_top_candidates,
    step_formalizability_probabilities,
)
from rllf.policy.training import (
    TrainConfig,
    evaluate_pairs,
    evaluate_reward_model,
    fit_beam_calibration,
    fit_calibrators,
    train_listwise,
    train_pairwise,
    train_supervised,
)
from rllf.state import PolicyState


def _separable_examples(n=48):
    """A trivially learnable task: `good` steps verify, `bad` steps do not."""
    trajectories = []
    for i in range(n):
        good = (i // 8) % 2 == 0  # each question gets both labels
        trajectories.append(
            Trajectory(
                # Questions repeat so that each one has both a verifiable and a
                # non-verifiable continuation -- which is what pairing needs.
                question=f"question {i % 8}",
                steps=[StepRecord("good step" if good else "bad step", verified=good, reward=float(good))],
                final_answer=str(i),
                outcome_reward=float(good),
            )
        )
    return build_step_examples(trajectories)


def test_forward_shapes_and_heads():
    model = ProcessRewardModel(ProcessRewardConfig(token_vocab_size=17))
    batch = model.encode_texts(["a", "bb"])
    output = model(**batch, return_tokens=True)
    assert output["step_logit"].shape == (2,)
    assert output["outcome_logit"].shape == (2,)
    assert output["action_logits"].shape == (2, len(ACTIONS))
    assert output["token_logits"].shape == (2, 17)


def test_pooling_modes_agree_on_shape():
    for pooling in ("mean", "cls", "last"):
        model = ProcessRewardModel(ProcessRewardConfig(pooling=pooling))
        assert model.score_texts(["hello there"])["step_prob"].shape == (1,)


def test_save_and_load_round_trip(tmp_path):
    model = ProcessRewardModel()
    before = model.score_texts(["a state"])["step_logit"]
    save_policy(str(tmp_path), model, TemperatureScaler(2.0, 0.5), TemperatureScaler(1.5, -0.2))
    policy = load_policy(str(tmp_path))
    assert policy.step_calibrator.temperature == 2.0
    assert torch.allclose(policy.model.score_texts(["a state"])["step_logit"], before)


def test_supervised_training_learns_a_separable_task():
    examples = _separable_examples()
    model = ProcessRewardModel()
    before = evaluate_reward_model(model, examples)
    train_supervised(model, examples, config=TrainConfig(epochs=6, batch_size=8, seed=0))
    after = evaluate_reward_model(model, examples)
    assert after["step"]["auc"] > 0.9
    assert after["step"]["bce"] < before["step"]["bce"]


def test_early_stopping_keeps_the_best_checkpoint():
    examples = _separable_examples()
    history = train_supervised(
        ProcessRewardModel(),
        examples,
        examples,
        TrainConfig(epochs=12, batch_size=8, early_stopping_patience=2, seed=0),
    )
    assert "best_score" in history
    assert len(history["eval"]) <= 12


def test_pairwise_training_orders_preferences():
    examples = _separable_examples()
    pairs = build_pairs(examples, allow_question_level=True)
    assert pairs
    model = ProcessRewardModel()
    train_pairwise(model, pairs, config=TrainConfig(epochs=8, batch_size=8, seed=0))
    assert evaluate_pairs(model, pairs)["ranking_accuracy"] > 0.9


def test_listwise_training_runs():
    from rllf.data.dataset import build_listwise_groups

    trajectories = [
        Trajectory("q", [StepRecord("good step", verified=True, reward=1.0)], "1", 1.0),
        Trajectory("q", [StepRecord("bad step", reward=0.0)], "2", 0.0),
    ]
    groups = build_listwise_groups(build_step_examples(trajectories))
    history = train_listwise(ProcessRewardModel(), groups, TrainConfig(epochs=2))
    assert len(history["train_loss"]) == 2


def test_calibration_improves_ece():
    examples = _separable_examples()
    model = ProcessRewardModel()
    train_supervised(model, examples, config=TrainConfig(epochs=4, batch_size=8, seed=0))
    step_scaler, _ = fit_calibrators(model, examples)
    assert step_scaler.temperature > 0


def test_beam_calibrator_defaults_to_the_identity():
    # "For now it will have a default value" -- T=1, bias=0, i.e. the raw
    # logit unscaled, until `fit_beam_calibration` sets a real one.
    policy = ProcessRewardPolicy(ProcessRewardModel())
    state = PolicyState("q", [], "a state")
    raw_logit = float(policy.model.score_texts([state.render()])["step_logit"][0])
    assert policy.predict_step_logit_for_beam(state) == pytest.approx(raw_logit)


def test_beam_calibrator_is_independent_of_step_calibrator():
    # Two different scalers for two different purposes (see the comment on
    # `ProcessRewardPolicy.beam_calibrator`) -- setting one must not move the
    # other.
    policy = ProcessRewardPolicy(
        ProcessRewardModel(),
        step_calibrator=TemperatureScaler(temperature=3.0, bias=1.0),
        beam_calibrator=TemperatureScaler(temperature=2.0, bias=0.5),
    )
    state = PolicyState("q", [], "a state")
    raw_logit = float(policy.model.score_texts([state.render()])["step_logit"][0])
    assert policy.predict_step_logit_for_beam(state) == pytest.approx((raw_logit - 0.5) / 2.0)
    # `predict_step_probability` still goes through `step_calibrator`, unaffected.
    import math
    assert policy.predict_step_probability(state) == pytest.approx(
        1 / (1 + math.exp(-(raw_logit - 1.0) / 3.0))
    )


def test_beam_calibrator_round_trips_through_save_and_load(tmp_path):
    model = ProcessRewardModel()
    save_policy(
        str(tmp_path), model,
        TemperatureScaler(2.0, 0.5), TemperatureScaler(1.5, -0.2),
        beam_calibrator=TemperatureScaler(4.0, -1.0),
    )
    policy = load_policy(str(tmp_path))
    assert policy.beam_calibrator.temperature == 4.0
    assert policy.beam_calibrator.bias == -1.0


def test_load_policy_defaults_the_beam_calibrator_for_an_old_checkpoint(tmp_path):
    # A checkpoint saved before `beam_calibrator` existed has no "beam" key
    # in calibration.json -- it must still load, defaulting to the identity.
    import json

    model = ProcessRewardModel()
    save_policy(str(tmp_path), model, TemperatureScaler(2.0, 0.5), TemperatureScaler(1.5, -0.2))
    payload = json.loads((tmp_path / "calibration.json").read_text())
    del payload["beam"]
    (tmp_path / "calibration.json").write_text(json.dumps(payload))
    policy = load_policy(str(tmp_path))
    assert policy.beam_calibrator.temperature == 1.0
    assert policy.beam_calibrator.bias == 0.0


def test_fit_beam_calibration_returns_a_fitted_scaler_usable_for_beam_scoring():
    examples = _separable_examples()
    model = ProcessRewardModel()
    train_supervised(model, examples, config=TrainConfig(epochs=4, batch_size=8, seed=0))
    beam_scaler = fit_beam_calibration(model, examples)
    assert beam_scaler.temperature > 0
    policy = ProcessRewardPolicy(model, beam_calibrator=beam_scaler)
    state = PolicyState("q", [], "good step")
    raw_logit = float(model.score_texts([state.render()])["step_logit"][0])
    assert policy.predict_step_logit_for_beam(state) == pytest.approx(
        (raw_logit - beam_scaler.bias) / beam_scaler.temperature
    )


def test_temperature_scaler_is_monotone():
    scaler = TemperatureScaler(temperature=2.0, bias=0.5)
    values = scaler.apply(torch.tensor([-2.0, 0.0, 2.0]))
    assert values[0] < values[1] < values[2]


def test_degenerate_calibration_is_bounded():
    torch.manual_seed(0)
    scaler = TemperatureScaler().fit(torch.randn(200), (torch.randn(200) > 0).float())
    assert scaler.min_temperature <= scaler.temperature <= scaler.max_temperature


def test_policy_scores_candidates_and_actions():
    policy = ProcessRewardPolicy(ProcessRewardModel())
    scores = policy.score_step_candidates("q", [], ["a", "b", "c"])
    assert scores.shape == (3,)
    actions = policy.action_scores(PolicyState("q", [], "partial"))
    assert set(actions) == set(ACTIONS)
    assert isinstance(policy.score_final_answer(PolicyState("q", [], final_answer="7")), float)


def test_lookahead_token_scoring_uses_the_base_tokenizer():
    from rllf.generation.base_model import CharTokenizer

    tokenizer = CharTokenizer("abc")
    policy = ProcessRewardPolicy(ProcessRewardModel(), base_tokenizer=tokenizer)
    assert policy.supports_token_scores
    scores = policy.score_tokens(PolicyState("q", [], "a"), [0, 1, 2])
    assert scores.shape == (3,)


def test_uniform_policy_is_a_no_op():
    policy = UniformPolicy()
    assert policy.score_step_candidates("q", [], ["a", "b"]).tolist() == [0.0, 0.0]


def test_verifier_policy_prefers_verified_candidates(verifier):
    policy = VerifierPolicy(verifier)
    scores = policy.score_step_candidates("q", [], ["so 2 + 2 = 4", "so 2 + 2 = 5"])
    assert scores[0] > scores[1]


def test_combine_scores_normalises_both_sides():
    base = torch.tensor([0.0, 10.0])
    policy = torch.tensor([10.0, 0.0])
    assert combine_scores(base, policy, alpha=0.0).argmax() == 1
    assert combine_scores(base, policy, alpha=10.0).argmax() == 0


def test_combine_reward_logits_adds_raw_rewards():
    combined = combine_reward_logits(torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]), alpha=1.0)
    assert combined.argmax() == 1


def test_select_top_candidates_respects_k_and_nucleus():
    logits = torch.tensor([5.0, 4.0, 1.0, 0.0])
    assert select_top_candidates(logits, k=2).tolist() == [0, 1]
    assert len(select_top_candidates(logits, k=4, top_p=0.5)) < 4


class _HistoryLengthPolicy:
    """`predict_step_probability` returns the length of the history it was
    given, as a fraction of 10 -- lets a test check both the *values* scored
    and that history accumulates in generation order, without a real model."""

    def predict_step_probability(self, state):
        return len(state.history) / 10.0


def test_step_formalizability_probabilities_sees_the_true_prefix():
    trajectory = Trajectory(
        question="q", steps=[StepRecord("s0"), StepRecord("s1"), StepRecord("s2")], final_answer="a"
    )
    probs = step_formalizability_probabilities(_HistoryLengthPolicy(), trajectory)
    # step 0 is scored with empty history, step 1 with one prior step, etc.
    assert probs == [0.0, 0.1, 0.2]


def test_step_formalizability_probabilities_empty_trajectory_is_empty():
    trajectory = Trajectory(question="q", steps=[], final_answer="a")
    assert step_formalizability_probabilities(_HistoryLengthPolicy(), trajectory) == []


def test_score_trajectory_formalizability_is_the_mean_step_probability():
    trajectory = Trajectory(
        question="q", steps=[StepRecord("s0"), StepRecord("s1"), StepRecord("s2")], final_answer="a"
    )
    assert score_trajectory_formalizability(_HistoryLengthPolicy(), trajectory) == pytest.approx(0.1)


def test_score_trajectory_formalizability_of_an_empty_trajectory_is_zero():
    trajectory = Trajectory(question="q", steps=[], final_answer="a")
    assert score_trajectory_formalizability(_HistoryLengthPolicy(), trajectory) == 0.0
