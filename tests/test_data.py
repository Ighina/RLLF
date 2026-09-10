import json

from rllf.data.dataset import (
    IGNORE,
    ProcessRewardDataset,
    build_listwise_groups,
    build_pairs,
    build_step_examples,
)
from rllf.data.trajectory import (
    StepRecord,
    Trajectory,
    load_trajectories,
    save_trajectories,
    split_by_question,
)
from rllf.policy.encoder import HashingTokenizer
from rllf.protocol import STEP_TOKEN


def _trajectories():
    return [
        Trajectory("q1", [StepRecord("3*4=12", verified=True, reward=1.0), StepRecord("12+1=14", reward=0.0)], "14", 0.0),
        Trajectory("q1", [StepRecord("3*4=13", reward=0.0)], "13", 0.0),
        Trajectory("q2", [StepRecord("2+2=4", verified=True, reward=1.0)], "4", 1.0),
    ]


def test_trajectory_round_trips_through_json(tmp_path):
    path = tmp_path / "t.jsonl"
    save_trajectories(path, _trajectories())
    loaded = load_trajectories(path)
    assert [t.question for t in loaded] == ["q1", "q1", "q2"]
    assert loaded[0].step_rewards == [1.0, 0.0]
    assert json.loads(path.read_text().splitlines()[0])["step_rewards"] == [1.0, 0.0]


def test_trajectory_accepts_the_plain_dict_schema():
    traj = Trajectory.from_dict(
        {"question": "q", "steps": ["a", "b"], "step_rewards": [1, 0], "final_answer": "3", "outcome_reward": 1}
    )
    assert traj.step_rewards == [1.0, 0.0]
    assert traj.steps[0].verified and not traj.steps[1].verified


def test_from_text_uses_the_step_protocol():
    traj = Trajectory.from_text("q", f"a {STEP_TOKEN} b {STEP_TOKEN} Final answer: 7")
    assert traj.step_texts == ["a", "b"]
    assert traj.final_answer == "Final answer: 7"


def test_fully_verified_and_history():
    traj = _trajectories()[0]
    assert not traj.fully_verified
    assert traj.history_at(1) == ["3*4=12"]
    assert _trajectories()[2].fully_verified


def test_split_by_question_does_not_leak():
    trajectories = _trajectories()
    train, held_out = split_by_question(trajectories, eval_fraction=0.5, seed=0)
    assert {t.question for t in train} & {t.question for t in held_out} == set()
    assert len(train) + len(held_out) == len(trajectories)


def test_build_step_examples_labels_and_masks():
    examples = build_step_examples(_trajectories())
    steps = [e for e in examples if not e.is_final]
    finals = [e for e in examples if e.is_final]
    assert [e.step_reward for e in steps] == [1.0, 0.0, 0.0, 1.0]
    assert all(e.step_reward == IGNORE for e in finals)
    assert finals[0].outcome_reward == 0.0


def test_example_text_matches_the_decoder_rendering():
    example = build_step_examples(_trajectories())[1]
    assert example.text.startswith("Question: q1")
    assert example.text.endswith("12+1=14")
    assert STEP_TOKEN in example.text  # history is rendered with boundaries


def test_pairs_prefer_the_same_prefix():
    pairs = build_pairs(build_step_examples(_trajectories()))
    assert ("3*4=12", "3*4=13") in [(p.preferred, p.rejected) for p in pairs]
    assert all(p.level == "prefix" for p in pairs)


def test_pairs_are_deduplicated():
    trajectories = _trajectories() + _trajectories()
    pairs = build_pairs(build_step_examples(trajectories))
    keys = [(p.preferred, p.rejected, tuple(p.history)) for p in pairs]
    assert len(keys) == len(set(keys))


def test_listwise_groups_need_mixed_labels():
    groups = build_listwise_groups(build_step_examples(_trajectories()))
    assert groups
    for group in groups:
        assert len({e.step_reward for e in group}) > 1


def test_collate_masks_missing_labels():
    examples = build_step_examples(_trajectories())
    dataset = ProcessRewardDataset(examples, HashingTokenizer())
    batch = dataset.collate(examples)
    assert batch["input_ids"].shape[0] == len(examples)
    assert batch["step_mask"].sum() == 4  # the three final-answer states are masked
    assert batch["outcome_mask"].sum() == len(examples)
    assert set(batch["step_labels"].tolist()) <= {0.0, 1.0}
