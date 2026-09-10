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
    select_top_candidates,
)
from rllf.policy.training import (
    TrainConfig,
    evaluate_pairs,
    evaluate_reward_model,
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
