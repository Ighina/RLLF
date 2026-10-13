"""Forked trajectory trees: the self-labelled training data for retraining the
step classifier *within* a problem (see `forked-tree-classifier-retraining.md`).

`experiments_v1.md` §10.1-§10.2 established that a classifier fitted on
FaithformBench's one-trajectory-per-problem steps learns problem difficulty
and an equation-density proxy, not "which of *these* competing steps will
prove", and that its step-level AUC against real Lean on our own generator's
steps is near chance. The fix is data the classifier has never had: several
solutions to the *same* problem, produced by the model we actually decode,
each step labelled by the real Lean pipeline.

A `ForkedTree` is one problem's worth of that data. Trajectory 0 is sampled
from scratch; each later trajectory *forks* from an existing one at a random
step boundary -- it reuses the prefix `steps[:fork_index]`, samples a fresh
step there, and continues to completion. That single structure feeds both
planned experiments:

* **full-trajectory rerank** (experiment 1): the complete trajectories, for
  an outcome-style ranker over whole solutions and a prefix-level process
  reward;
* **constrained beam search** (experiment 2): at every fork point, two
  sibling steps that continue a byte-identical prefix -- exactly the
  `(prefix, candidate step)` comparison the beam search makes at a `<step>`
  boundary -- each with its own Lean verdict *and* the fate of its
  continuation.

Shared prefixes are verified once: `unique_steps` keys every `(question,
history, step)` by content, so a fork costs only its new steps.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .dataset import IGNORE, PairExample, StepExample
from .trajectory import StepRecord, Trajectory


def step_key(question: str, history: Sequence[str], step: str) -> str:
    """Content address of one `(question, history, step)` state -- the unit the
    real-Lean pipeline scores (`render_step_claim` sees exactly these three),
    so two trajectories sharing a prefix share every key along it."""
    payload = json.dumps([question.strip(), [h.strip() for h in history], step.strip()], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class ForkedTrajectory:
    steps: List[str]
    #: Raw final segment (`<final> ...` / `\\boxed{...}` text), "" if the
    #: trajectory ended without one.
    final_answer: str = ""
    #: Index of the trajectory this one forked from; None for a root sample.
    parent: Optional[int] = None
    #: `steps[:fork_index]` is shared with `parent`; `steps[fork_index]` is the
    #: fresh sibling of `parent.steps[fork_index]`. None for a root sample.
    fork_index: Optional[int] = None
    #: Outcome reward from the outcome checker (1.0 / 0.0), None if unchecked.
    correct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def history_at(self, index: int) -> List[str]:
        return self.steps[:index]


@dataclass
class ForkedTree:
    question: str
    reference: Optional[str]
    trajectories: List[ForkedTrajectory]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ForkedTree":
        return cls(
            question=payload["question"],
            reference=payload.get("reference"),
            trajectories=[ForkedTrajectory(**t) for t in payload.get("trajectories", [])],
            metadata=payload.get("metadata", {}) or {},
        )

    # -- views ------------------------------------------------------------
    def unique_steps(self) -> Dict[str, Tuple[List[str], str]]:
        """Every distinct `(history, step)` state in the tree, keyed by
        `step_key` -- the Lean work list. Shared prefixes appear once."""
        out: Dict[str, Tuple[List[str], str]] = {}
        for traj in self.trajectories:
            for i, step in enumerate(traj.steps):
                key = step_key(self.question, traj.steps[:i], step)
                if key not in out:
                    out[key] = (traj.steps[:i], step)
        return out

    def step_keys(self, traj: ForkedTrajectory) -> List[str]:
        return [step_key(self.question, traj.steps[:i], s) for i, s in enumerate(traj.steps)]

    def sibling_pairs(self) -> List["SiblingPair"]:
        """One pair per fork: the parent's step at `fork_index` and the fresh
        step that replaced it, both continuing the same prefix. Identical
        siblings (the sampler re-drew the same text) are skipped -- there is
        nothing to rank."""
        pairs: List[SiblingPair] = []
        for child_index, child in enumerate(self.trajectories):
            if child.parent is None or child.fork_index is None:
                continue
            parent = self.trajectories[child.parent]
            k = child.fork_index
            if k >= len(parent.steps) or k >= len(child.steps):
                continue
            if parent.steps[k].strip() == child.steps[k].strip():
                continue
            pairs.append(
                SiblingPair(
                    question=self.question,
                    prefix=list(parent.steps[:k]),
                    a_index=child.parent, a_step=parent.steps[k],
                    b_index=child_index, b_step=child.steps[k],
                )
            )
        return pairs


@dataclass
class SiblingPair:
    """Two steps continuing the same prefix, one from each side of a fork."""

    question: str
    prefix: List[str]
    a_index: int
    a_step: str
    b_index: int
    b_step: str

    @property
    def a_key(self) -> str:
        return step_key(self.question, self.prefix, self.a_step)

    @property
    def b_key(self) -> str:
        return step_key(self.question, self.prefix, self.b_step)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


def save_trees(path: str | Path, trees: Iterable[ForkedTree]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for tree in trees:
            handle.write(json.dumps(tree.to_dict(), ensure_ascii=False) + "\n")
    return path


def iter_trees(path: str | Path) -> Iterator[ForkedTree]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield ForkedTree.from_dict(json.loads(line))


def load_trees(path: str | Path) -> List[ForkedTree]:
    return list(iter_trees(path))


# --------------------------------------------------------------------------
# Labels -> training data
# --------------------------------------------------------------------------

#: `labels[step_key] -> {"proved": bool, "stage": str, ...}` -- the real-Lean
#: verdict file written by the verification script, one entry per unique step.
LeanLabels = Dict[str, Dict[str, Any]]


def trajectory_true_rate(tree: ForkedTree, traj: ForkedTrajectory, labels: LeanLabels) -> Optional[float]:
    """Fraction of the trajectory's steps the real pipeline proved; None if
    any step is still unlabelled (a partially-verified tree is not usable
    for whole-trajectory targets)."""
    keys = tree.step_keys(traj)
    if not keys or any(k not in labels for k in keys):
        return None
    return sum(1.0 for k in keys if labels[k]["proved"]) / len(keys)


def to_trajectory(tree: ForkedTree, index: int, labels: LeanLabels) -> Trajectory:
    """A labelled `ForkedTrajectory` as the codebase's standard `Trajectory`,
    so every existing consumer (`build_step_examples`, the policy scorers,
    `full_trajectory_rerank`'s ablation) reads it unchanged."""
    traj = tree.trajectories[index]
    records = []
    for key, step in zip(tree.step_keys(traj), traj.steps):
        verdict = labels.get(key)
        proved = bool(verdict and verdict.get("proved"))
        records.append(
            StepRecord(
                text=step,
                formalization=(verdict or {}).get("lean_statement"),
                verified=proved,
                reward=1.0 if proved else (0.0 if verdict else IGNORE),
                status=(verdict or {}).get("stage", "unverified"),
                error=(verdict or {}).get("error"),
                diagnostics={"labelled": verdict is not None},
            )
        )
    return Trajectory(
        question=tree.question,
        steps=records,
        final_answer=traj.final_answer,
        outcome_reward=float(traj.correct) if traj.correct is not None else 0.0,
        reference=tree.reference,
        metadata={
            "source": "forked_tree", "tree_index": index, "parent": traj.parent,
            "fork_index": traj.fork_index, "correct_known": traj.correct is not None,
            **tree.metadata, **traj.metadata,
        },
    )


def build_prm_examples(trees: Sequence[ForkedTree], labels: LeanLabels) -> List[StepExample]:
    """Per-step `(question, history, step)` states with the real-Lean verdict
    as `step_reward` and the trajectory's correctness as `outcome_reward`.

    Shared prefix steps are emitted once per tree (they are the same state,
    with the same label, whichever trajectory they sit on). `group_key`
    (`StepExample`) then already groups siblings for `build_pairs` /
    `build_listwise_groups`, and `metadata["problem_id"]` groups the whole
    problem for problem-level contrasts.
    """
    examples: List[StepExample] = []
    for tree_id, tree in enumerate(trees):
        seen = set()
        for traj_id, traj in enumerate(tree.trajectories):
            for i, (key, step) in enumerate(zip(tree.step_keys(traj), traj.steps)):
                if key in seen or key not in labels:
                    continue
                seen.add(key)
                examples.append(
                    StepExample(
                        question=tree.question,
                        history=traj.steps[:i],
                        step=step,
                        step_reward=1.0 if labels[key]["proved"] else 0.0,
                        outcome_reward=float(traj.correct) if traj.correct is not None else IGNORE,
                        step_index=i,
                        trajectory_id=traj_id,
                        metadata={"problem_id": tree_id, "step_key": key, "stage": labels[key].get("stage")},
                    )
                )
    return examples


def build_sibling_pairs(
    trees: Sequence[ForkedTree], labels: LeanLabels, target: str = "proved"
) -> List[PairExample]:
    """Prefix-level preference pairs from the fork points.

    `target="proved"`: prefer the sibling whose own step proved (the
    beam-search boundary decision, same target the classifier has always
    had). `target="continuation"`: prefer the sibling whose continuation had
    the higher true formalization rate (a value-style target). Ties and
    unlabelled sides are dropped.
    """
    pairs: List[PairExample] = []
    for tree in trees:
        for pair in tree.sibling_pairs():
            if target == "proved":
                a = labels.get(pair.a_key)
                b = labels.get(pair.b_key)
                if a is None or b is None or bool(a["proved"]) == bool(b["proved"]):
                    continue
                a_score, b_score = float(a["proved"]), float(b["proved"])
            elif target == "continuation":
                a_rate = trajectory_true_rate(tree, tree.trajectories[pair.a_index], labels)
                b_rate = trajectory_true_rate(tree, tree.trajectories[pair.b_index], labels)
                if a_rate is None or b_rate is None or a_rate == b_rate:
                    continue
                a_score, b_score = a_rate, b_rate
            else:
                raise ValueError(f"unknown target {target!r}")
            preferred, rejected = (pair.a_step, pair.b_step) if a_score > b_score else (pair.b_step, pair.a_step)
            pairs.append(PairExample(tree.question, list(pair.prefix), preferred, rejected, "prefix"))
    return pairs


def build_problem_pairs(
    examples: Sequence[StepExample], max_pairs_per_problem: int = 16, seed: int = 0
) -> List[PairExample]:
    """Within-problem `(proved, unproved)` step pairs drawn across *all* of a
    problem's trajectories, each side keeping its own history
    (`PairExample.level == "question"`).

    `rllf.data.dataset.build_pairs` only emits question-level pairs for
    problems that have no same-prefix pair at all; here every problem
    contributes, because the whole point of this data is the within-problem
    contrast (`experiments_v1.md` §10.1: the pooled score/true-rate
    correlation is positive while the within-problem one is negative --
    only pairs that hold the problem fixed can correct that).
    """
    rng = random.Random(seed)
    by_problem: Dict[Any, List[StepExample]] = defaultdict(list)
    for e in examples:
        if not e.is_final and e.step_reward != IGNORE:
            by_problem[e.metadata.get("problem_id", e.question)].append(e)
    pairs: List[PairExample] = []
    for group in by_problem.values():
        positives = [e for e in group if e.step_reward > 0.5]
        negatives = [e for e in group if e.step_reward <= 0.5]
        combos = [(p, n) for p in positives for n in negatives]
        rng.shuffle(combos)
        for pos, neg in combos[:max_pairs_per_problem]:
            pairs.append(PairExample(pos.question, list(pos.history), pos.step, neg.step, "question"))
    return pairs


def split_trees(trees: Sequence[ForkedTree], eval_fraction: float = 0.2, seed: int = 0) -> Tuple[List[ForkedTree], List[ForkedTree]]:
    """By-problem split (a tree *is* a problem, so this is leak-free by
    construction). Returns `(fit, held_out)`."""
    order = list(range(len(trees)))
    random.Random(seed).shuffle(order)
    n_eval = int(round(len(trees) * eval_fraction))
    held = set(order[:n_eval])
    return [t for i, t in enumerate(trees) if i not in held], [t for i, t in enumerate(trees) if i in held]


@dataclass
class TrajectoryGroup:
    """One problem's competing full trajectories with their whole-trajectory
    targets -- the outcome-ranker's training unit (experiment 1)."""

    question: str
    trajectories: List[Trajectory]
    true_rates: List[float]
    correct: List[Optional[float]]


def build_trajectory_groups(trees: Sequence[ForkedTree], labels: LeanLabels, min_size: int = 2) -> List[TrajectoryGroup]:
    groups: List[TrajectoryGroup] = []
    for tree in trees:
        members, rates, correct = [], [], []
        for index, traj in enumerate(tree.trajectories):
            rate = trajectory_true_rate(tree, traj, labels)
            if rate is None:
                continue
            members.append(to_trajectory(tree, index, labels))
            rates.append(rate)
            correct.append(traj.correct)
        if len(members) >= min_size and len(set(rates)) > 1:
            groups.append(TrajectoryGroup(tree.question, members, rates, correct))
    return groups


def summarize(trees: Sequence[ForkedTree], labels: Optional[LeanLabels] = None) -> Dict[str, Any]:
    """Counts a run script can print: trajectories, unique vs. total steps
    (the fork saving), fork pairs, and -- with labels -- coverage and rates."""
    n_traj = sum(len(t.trajectories) for t in trees)
    total_steps = sum(len(traj.steps) for t in trees for traj in t.trajectories)
    unique = sum(len(t.unique_steps()) for t in trees)
    pairs = [p for t in trees for p in t.sibling_pairs()]
    out: Dict[str, Any] = {
        "problems": len(trees), "trajectories": n_traj, "total_steps": total_steps,
        "unique_steps": unique, "lean_saving": 1 - unique / total_steps if total_steps else 0.0,
        "sibling_pairs": len(pairs),
        "reached_answer": sum(1 for t in trees for traj in t.trajectories if traj.final_answer),
        "correct": sum(1 for t in trees for traj in t.trajectories if traj.correct),
    }
    if labels is not None:
        keys = [k for t in trees for k in t.unique_steps()]
        labelled = [k for k in keys if k in labels]
        out["labelled_steps"] = len(labelled)
        out["proved_rate"] = (
            sum(1 for k in labelled if labels[k]["proved"]) / len(labelled) if labelled else None
        )
        by_target = defaultdict(int)
        for target in ("proved", "continuation"):
            by_target[target] = len(build_sibling_pairs(trees, labels, target))
        out["usable_pairs"] = dict(by_target)
        out["trajectory_groups"] = len(build_trajectory_groups(trees, labels))
    return out
