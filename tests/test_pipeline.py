from rllf.data.dataset import build_step_examples
from rllf.data.synthetic import SyntheticBaseModel
from rllf.data.trajectory import load_trajectories
from rllf.evaluation.protocol import report_table, run_evaluation
from rllf.pipeline import (
    GeneratorSpec,
    TrajectoryCollector,
    annotate_trajectories,
    dataset_summary,
    load_problems,
)
from rllf.verification.lean_verifier import MockStepVerifier
from rllf.verification.outcome_checker import default_outcome_checker


def _collector(problems, verifier, outcome_checker, error_rates=(0.2, 0.6)):
    generators = [
        GeneratorSpec(
            name=f"gen-{rate}",
            model=SyntheticBaseModel(problems, error_rate=rate, seed=index),
            num_samples=2,
            max_steps=4,
            seed=index * 10,
        )
        for index, rate in enumerate(error_rates)
    ]
    return TrajectoryCollector(generators, verifier, outcome_checker)


def test_collect_produces_labelled_trajectories(problems, verifier, outcome_checker):
    trajectories = _collector(problems, verifier, outcome_checker).collect(problems)
    assert len(trajectories) == len(problems) * 2 * 2
    assert all(t.reference for t in trajectories)
    assert all(len(t.steps) == 2 for t in trajectories)
    # The labels must not be degenerate, or there is nothing to learn.
    rewards = {s.reward for t in trajectories for s in t.steps}
    assert rewards == {0.0, 1.0}
    assert 0.0 < sum(t.outcome_reward for t in trajectories) < len(trajectories)


def test_multiple_generators_are_recorded(problems, verifier, outcome_checker):
    trajectories = _collector(problems, verifier, outcome_checker).collect(problems)
    assert len(dataset_summary(trajectories)["generators"]) == 2


def test_collect_writes_jsonl(tmp_path, problems, verifier, outcome_checker):
    path = tmp_path / "traj.jsonl"
    _collector(problems, verifier, outcome_checker).collect(problems, output_path=path)
    assert len(load_trajectories(path)) == len(problems) * 2 * 2


def test_summary_reports_failure_modes(problems, verifier, outcome_checker):
    summary = dataset_summary(_collector(problems, verifier, outcome_checker).collect(problems))
    assert summary["n_questions"] == len(problems)
    assert summary["n_steps"] > 0
    assert set(summary["status_counts"]) <= {"verified", "lean_check_failed", "formalization_failed", "skipped"}


def test_examples_carry_both_reward_signals(problems, verifier, outcome_checker):
    trajectories = _collector(problems, verifier, outcome_checker).collect(problems)
    examples = build_step_examples(trajectories)
    assert any(e.is_final for e in examples)
    assert any(e.step_reward == 1.0 for e in examples)
    assert any(e.step_reward == 0.0 for e in examples)


def test_annotate_trajectories_labels_existing_data(problems, verifier, outcome_checker):
    trajectories = _collector(problems, None, None).collect(problems)
    assert all(s.status == "unverified" for t in trajectories for s in t.steps)
    labelled = annotate_trajectories(trajectories, verifier, outcome_checker)
    assert all(s.status != "unverified" for t in labelled for s in t.steps)


def test_load_problems(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"question": "q1", "answer": "3", "difficulty": "easy"}\n')
    problems = load_problems(path)
    assert problems[0].reference == "3"
    assert problems[0].metadata["difficulty"] == "easy"


def test_evaluation_protocol_runs_every_method(problems):
    verifier = MockStepVerifier()
    model = SyntheticBaseModel(problems, error_rate=0.5, seed=0)
    results = run_evaluation(
        problems, model, policy=None, verifier=verifier,
        outcome_checker=default_outcome_checker(), num_candidates=2, max_steps=4,
    )
    # Without a policy only the policy-free baselines run.
    assert set(results) == {"greedy", "sampling", "verification_only"}
    assert "outcome_accuracy" in results["greedy"].report
    assert report_table(results)


def test_oracle_policy_beats_the_base_model(problems):
    verifier = MockStepVerifier()
    model = SyntheticBaseModel(problems, error_rate=0.6, seed=0)
    results = run_evaluation(
        problems, model, policy=None, verifier=verifier,
        outcome_checker=default_outcome_checker(), num_candidates=6, max_steps=4,
        methods=["greedy", "oracle_policy"], include_oracle=True,
    )
    assert results["oracle_policy"].report["step_verification_rate"] >= results["greedy"].report[
        "step_verification_rate"
    ]


def test_policy_gradient_step_runs(problems, verifier, outcome_checker):
    from rllf.policy.model import ProcessRewardModel
    from rllf.policy.training import PolicyGradientConfig, PolicyGradientTrainer

    trainer = PolicyGradientTrainer(
        ProcessRewardModel(),
        SyntheticBaseModel(problems, error_rate=0.5, seed=0),
        verifier,
        outcome_checker,
        PolicyGradientConfig(num_candidates=3, max_steps=4, device="cpu"),
    )
    stats = trainer.step([(p.question, p.reference) for p in problems[:3]])
    assert stats["n_actions"] > 0
    assert 0.0 <= stats["outcome_accuracy"] <= 1.0


def test_policy_gradient_returns_discount_and_add_outcome(problems, verifier, outcome_checker):
    from rllf.policy.model import ProcessRewardModel
    from rllf.policy.training import PolicyGradientConfig, PolicyGradientTrainer, Rollout

    trainer = PolicyGradientTrainer(
        ProcessRewardModel(), SyntheticBaseModel(problems, seed=0), verifier, outcome_checker,
        PolicyGradientConfig(gamma=0.5, outcome_weight=2.0, device="cpu"),
    )
    rollout = Rollout(trajectory=None, log_probs=[None, None], step_rewards=[1.0, 1.0], outcome_reward=1.0)
    assert trainer.returns(rollout) == [1.5 + 2.0, 1.0 + 2.0]
