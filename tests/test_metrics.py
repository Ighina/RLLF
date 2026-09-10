from rllf.data.trajectory import StepRecord, Trajectory
from rllf.evaluation.metrics import (
    accuracy_by_difficulty,
    binary_metrics,
    expected_calibration_error,
    format_report,
    formalization_success_rate,
    fully_verified_rate,
    outcome_accuracy,
    pairwise_ranking_accuracy,
    roc_auc,
    step_verification_rate,
    trajectory_report,
)


def _trajectories():
    return [
        Trajectory(
            "q1",
            [StepRecord("a", verified=True, reward=1.0, status="verified"),
             StepRecord("b", reward=0.0, status="lean_check_failed")],
            "1", 0.0, metadata={"difficulty": "easy", "lean_calls": 2.0},
        ),
        Trajectory(
            "q2",
            [StepRecord("c", verified=True, reward=1.0, status="verified")],
            "2", 1.0, metadata={"difficulty": "hard", "lean_calls": 1.0},
        ),
    ]


def test_step_verification_rate_is_over_steps_not_trajectories():
    assert step_verification_rate(_trajectories()) == 2 / 3


def test_fully_verified_and_outcome():
    assert fully_verified_rate(_trajectories()) == 0.5
    assert outcome_accuracy(_trajectories()) == 0.5


def test_formalization_rate_counts_only_formalization_failures():
    trajectories = _trajectories()
    trajectories[0].steps[1].status = "formalization_failed"
    assert formalization_success_rate(trajectories) == 2 / 3


def test_empty_inputs_do_not_divide_by_zero():
    assert step_verification_rate([]) == 0.0
    assert outcome_accuracy([]) == 0.0
    assert binary_metrics([], [])["auc"] == 0.5


def test_trajectory_report_pulls_cost_from_metadata():
    report = trajectory_report(_trajectories())
    assert report["avg_lean_calls"] == 1.5
    assert report["avg_steps"] == 1.5


def test_accuracy_by_difficulty():
    assert accuracy_by_difficulty(_trajectories()) == {"easy": 0.0, "hard": 1.0}


def test_roc_auc_matches_known_values():
    assert roc_auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == 0.75
    assert roc_auc([1.0, 1.0], [0, 1]) == 0.5  # all ties
    assert roc_auc([0.9, 0.1], [1, 0]) == 1.0


def test_expected_calibration_error():
    assert expected_calibration_error([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert expected_calibration_error([1.0, 1.0], [0.0, 0.0]) == 1.0


def test_binary_metrics_fields():
    metrics = binary_metrics([0.9, 0.2], [1.0, 0.0])
    assert metrics["accuracy"] == 1.0
    assert metrics["positive_rate"] == 0.5
    assert 0.0 < metrics["bce"] < 1.0


def test_pairwise_ranking_accuracy():
    assert pairwise_ranking_accuracy([1.0, 0.0], [0.0, 1.0]) == 0.5


def test_format_report_renders_a_table():
    table = format_report({"a": {"x": 1.0}, "b": {"x": 0.5}}, ["x"])
    assert "method" in table and "a" in table and "1.000" in table
