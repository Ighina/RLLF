"""End-to-end smoke test of the command line pipeline."""

import json

from train import main


def test_generate_train_evaluate(tmp_path):
    trajectories = tmp_path / "traj.jsonl"
    policy_dir = tmp_path / "prm"

    assert main([
        "generate", "--mock-verifier", "--synthetic", "8", "--samples-per-problem", "2",
        "--out", str(trajectories), "--cache", str(tmp_path / "cache.jsonl"),
    ]) == 0
    assert trajectories.exists()
    records = [json.loads(line) for line in trajectories.read_text().splitlines()]
    assert len(records) == 8 * 2 * 3  # problems x samples x generators
    assert {"question", "steps", "step_rewards", "outcome_reward"} <= set(records[0])

    assert main([
        "train", "--trajectories", str(trajectories), "--out", str(policy_dir),
        "--epochs", "2", "--batch-size", "8",
    ]) == 0
    assert (policy_dir / "model.pt").exists()
    assert (policy_dir / "config.json").exists()
    calibration = json.loads((policy_dir / "calibration.json").read_text())
    assert "temperature" in calibration["step"]

    assert main([
        "evaluate", "--mock-verifier", "--policy", str(policy_dir), "--synthetic", "4",
        "--num-candidates", "2", "--best-of", "2", "--cache", str(tmp_path / "cache.jsonl"),
    ]) == 0
