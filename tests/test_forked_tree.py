import random

import pytest

from rllf.data.dataset import IGNORE, build_listwise_groups, build_pairs
from rllf.data.forked_tree import (
    ForkedTrajectory,
    ForkedTree,
    build_prm_examples,
    build_sibling_pairs,
    build_trajectory_groups,
    load_trees,
    save_trees,
    step_key,
    summarize,
    trajectory_true_rate,
)
from rllf.generation.forked_sampling import ForkConfig, continue_trajectory, sample_forked_tree
from rllf.protocol import FINAL_TOKEN
from rllf.verification.outcome_checker import default_outcome_checker


class ScriptedStepModel:
    """Step-native double: at depth d it draws one of `options[d]`; at depth
    `len(options)` it emits `<final> <answer>`. Deterministic given `seed`."""

    def __init__(self, options, answer="42"):
        self.options = options
        self.answer = answer
        self.calls = 0

    def generate_step_candidates(self, question, history, num_candidates=1, seed=None, **kwargs):
        self.calls += 1
        depth = len(history)
        if depth >= len(self.options):
            return [f"{FINAL_TOKEN} {self.answer}"]
        rng = random.Random(seed)
        return [rng.choice(self.options[depth])]


OPTIONS = [["a1", "a2"], ["b1", "b2", "b3"], ["c1", "c2"]]


def test_continue_trajectory_stops_at_final():
    model = ScriptedStepModel(OPTIONS)
    steps, final, reason = continue_trajectory("q", [], model, ForkConfig(seed=1), random.Random(1))
    assert len(steps) == 3 and reason == "final" and final.startswith(FINAL_TOKEN)


def test_continue_trajectory_respects_max_steps_and_empty():
    model = ScriptedStepModel(OPTIONS)
    steps, final, reason = continue_trajectory("q", [], model, ForkConfig(max_steps=2), random.Random(0))
    assert len(steps) == 2 and final == "" and reason == "max_steps"

    class Empty:
        def generate_step_candidates(self, *a, **k):
            return [""]

    steps, final, reason = continue_trajectory("q", ["x"], Empty(), ForkConfig(), random.Random(0))
    assert steps == ["x"] and reason == "empty"


def test_sample_forked_tree_shares_prefixes_and_records_forks():
    model = ScriptedStepModel(OPTIONS)
    tree = sample_forked_tree("q", "42", model, default_outcome_checker(), ForkConfig(n_trajectories=4, seed=3))
    assert len(tree.trajectories) == 4
    root = tree.trajectories[0]
    assert root.parent is None and root.fork_index is None
    for child in tree.trajectories[1:]:
        assert child.parent is not None and child.fork_index is not None
        parent = tree.trajectories[child.parent]
        assert child.steps[: child.fork_index] == parent.steps[: child.fork_index]
        assert "identical_sibling" in child.metadata
    assert all(t.correct == 1.0 for t in tree.trajectories)
    assert all(t.metadata["stop_reason"] == "final" for t in tree.trajectories)


def test_unique_steps_dedupe_shared_prefix_and_keys_are_content_addressed():
    a = ForkedTrajectory(steps=["s1", "s2", "s3"], final_answer="<final> 1")
    b = ForkedTrajectory(steps=["s1", "s2", "x3"], final_answer="<final> 1", parent=0, fork_index=2)
    tree = ForkedTree("q", "1", [a, b])
    unique = tree.unique_steps()
    assert len(unique) == 4  # s1, s2 shared; s3 and x3 distinct
    assert step_key("q", ["s1", "s2"], "s3") in unique
    assert step_key("q", ["s1"], "s2") == step_key("q ", ["s1 "], " s2")


def test_sibling_pairs_skip_identical_and_out_of_range():
    a = ForkedTrajectory(steps=["s1", "s2"], final_answer="")
    same = ForkedTrajectory(steps=["s1", "s2"], final_answer="", parent=0, fork_index=1)
    diff = ForkedTrajectory(steps=["s1", "t2", "t3"], final_answer="", parent=0, fork_index=1)
    short = ForkedTrajectory(steps=["s1"], final_answer="", parent=0, fork_index=1)
    tree = ForkedTree("q", None, [a, same, diff, short])
    pairs = tree.sibling_pairs()
    assert len(pairs) == 1
    assert pairs[0].prefix == ["s1"] and {pairs[0].a_step, pairs[0].b_step} == {"s2", "t2"}


def _labelled_tree():
    a = ForkedTrajectory(steps=["s1", "s2", "s3"], final_answer="<final> 1", correct=1.0)
    b = ForkedTrajectory(steps=["s1", "x2", "x3"], final_answer="<final> 2", parent=0, fork_index=1, correct=0.0)
    tree = ForkedTree("q", "1", [a, b])
    labels = {
        step_key("q", [], "s1"): {"proved": True, "stage": "proved"},
        step_key("q", ["s1"], "s2"): {"proved": True, "stage": "proved"},
        step_key("q", ["s1", "s2"], "s3"): {"proved": False, "stage": "prove"},
        step_key("q", ["s1"], "x2"): {"proved": False, "stage": "formalize"},
        step_key("q", ["s1", "x2"], "x3"): {"proved": False, "stage": "prove"},
    }
    return tree, labels


def test_true_rate_and_prm_examples():
    tree, labels = _labelled_tree()
    assert trajectory_true_rate(tree, tree.trajectories[0], labels) == pytest.approx(2 / 3)
    assert trajectory_true_rate(tree, tree.trajectories[1], labels) == pytest.approx(1 / 3)
    del labels[step_key("q", ["s1", "x2"], "x3")]
    assert trajectory_true_rate(tree, tree.trajectories[1], labels) is None

    examples = build_prm_examples([tree], labels)
    assert len(examples) == 4  # shared s1 once; x3 unlabelled dropped
    by_step = {e.step: e for e in examples}
    assert by_step["s1"].step_reward == 1.0 and by_step["x2"].step_reward == 0.0
    assert by_step["s2"].outcome_reward == 1.0 and by_step["x2"].outcome_reward == 0.0
    # siblings share a group key -> the existing pair/list builders see them
    assert by_step["s2"].group_key == by_step["x2"].group_key
    assert len(build_pairs(examples, allow_question_level=False)) == 1
    assert len(build_listwise_groups(examples)) == 1


def test_sibling_pairs_by_target_and_trajectory_groups():
    tree, labels = _labelled_tree()
    proved = build_sibling_pairs([tree], labels, target="proved")
    assert len(proved) == 1 and proved[0].preferred == "s2" and proved[0].rejected == "x2"
    cont = build_sibling_pairs([tree], labels, target="continuation")
    assert len(cont) == 1 and cont[0].preferred == "s2"
    with pytest.raises(ValueError):
        build_sibling_pairs([tree], labels, target="nope")

    groups = build_trajectory_groups([tree], labels)
    assert len(groups) == 1
    assert groups[0].true_rates == pytest.approx([2 / 3, 1 / 3])
    assert groups[0].trajectories[0].outcome_reward == 1.0
    assert groups[0].trajectories[1].steps[1].reward == 0.0 and groups[0].trajectories[1].steps[1].status == "formalize"


def test_round_trip_and_summary(tmp_path):
    tree, labels = _labelled_tree()
    path = save_trees(tmp_path / "trees.jsonl", [tree])
    back = load_trees(path)
    assert back[0].to_dict() == tree.to_dict()
    s = summarize([tree], labels)
    assert s["total_steps"] == 6 and s["unique_steps"] == 5 and s["sibling_pairs"] == 1
    assert s["labelled_steps"] == 5 and s["proved_rate"] == pytest.approx(0.4)
    assert s["usable_pairs"] == {"proved": 1, "continuation": 1} and s["trajectory_groups"] == 1
    assert s["reached_answer"] == 2 and s["correct"] == 1


def test_unlabelled_step_reward_is_ignore_in_to_trajectory():
    from rllf.data.forked_tree import to_trajectory

    tree, labels = _labelled_tree()
    del labels[step_key("q", ["s1", "s2"], "s3")]
    traj = to_trajectory(tree, 0, labels)
    assert traj.steps[2].reward == IGNORE and traj.steps[2].diagnostics["labelled"] is False


def test_problem_pairs_and_split():
    from rllf.data.forked_tree import build_problem_pairs, split_trees

    tree, labels = _labelled_tree()
    examples = build_prm_examples([tree], labels)
    pairs = build_problem_pairs(examples, max_pairs_per_problem=10)
    # positives s1, s2 x negatives s3, x2, x3 -> 6 combos
    assert len(pairs) == 6 and all(p.level == "question" for p in pairs)
    assert len(build_problem_pairs(examples, max_pairs_per_problem=2)) == 2
    trees = [ForkedTree(f"q{i}", None, []) for i in range(10)]
    fit, held = split_trees(trees, 0.2, seed=0)
    assert len(fit) == 8 and len(held) == 2 and not {t.question for t in fit} & {t.question for t in held}


def test_train_trajectory_listwise_and_tree_eval():
    from rllf.evaluation.forked_tree_eval import evaluate_policy_on_trees, format_tree_report, spearman
    from rllf.policy.model import ProcessRewardModel
    from rllf.policy.scoring import ProcessRewardPolicy
    from rllf.policy.training import TrainConfig, train_trajectory_listwise

    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1], [1]) is None

    tree, labels = _labelled_tree()
    groups = build_trajectory_groups([tree], labels)
    model = ProcessRewardModel()
    history = train_trajectory_listwise(model, groups, TrainConfig(epochs=2), eval_groups=groups)
    assert len(history["train_loss"]) == 2 and history["eval"][-1]["n_groups"] == 1.0
    with pytest.raises(ValueError):
        train_trajectory_listwise(model, groups, TrainConfig(epochs=1), target="nope")

    report = evaluate_policy_on_trees(ProcessRewardPolicy(model, outcome_weight=0.0), [tree], labels)
    assert report["step"]["n"] == 5 and report["sibling_pairs"] == 1
    assert report["prm_pick"]["n_problems"] == 1 and report["prm_pick"]["oracle_true_rate"] == pytest.approx(2 / 3)
    assert report["orm_pick"]["random_true_rate"] == pytest.approx(0.5)
    text = format_tree_report({"m": report})
    assert "stepAUC" in text and "answer accuracy" in text and report["prm_pick"]["oracle_correct_rate"] == 1.0


def test_ranking_trainers_restore_best_epoch():
    from rllf.data.dataset import PairExample
    from rllf.policy.model import ProcessRewardModel
    from rllf.policy.training import TrainConfig, train_listwise, train_pairwise, train_trajectory_listwise

    tree, labels = _labelled_tree()
    pairs = build_sibling_pairs([tree], labels) + [PairExample("q", ["s1"], "s2", "x2", "prefix")]
    model = ProcessRewardModel()
    h = train_pairwise(model, pairs, pairs, TrainConfig(epochs=2, early_stopping_patience=1))
    assert "best_epoch" in h and h["select_by"] == "pair_ranking_accuracy" and len(h["eval"]) >= 1
    groups = build_listwise_groups(build_prm_examples([tree], labels))
    h = train_listwise(ProcessRewardModel(), groups, TrainConfig(epochs=2), eval_pairs=pairs)
    assert "best_epoch" in h and len(h["eval"]) == 2
    h = train_trajectory_listwise(ProcessRewardModel(), build_trajectory_groups([tree], labels), TrainConfig(epochs=2),
                                  eval_groups=build_trajectory_groups([tree], labels))
    assert h["select_by"] == "orm_pick_true_rate" and "best_epoch" in h
