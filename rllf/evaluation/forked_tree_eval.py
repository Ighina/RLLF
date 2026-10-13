"""Evaluating a policy on held-out `ForkedTree`s -- the metrics that
actually matter for decoding, measured on the decoded model's own steps
against real-Lean verdicts (the offline FaithformBench AUC did not predict
any of them: `experiments_v1.md` §10.2).

* `step_auc`         -- per-step probability vs. proved, all unique steps;
* `sibling_accuracy` -- at each fork, is the proved sibling scored above the
                        unproved one (the beam-search boundary decision);
* `prm_pick` / `orm_pick` -- per problem, the trajectory the policy would
                        select (mean step probability / outcome head on the
                        full trajectory): its true formalization rate vs.
                        the problem mean (random) and max (oracle), its
                        correctness, and the within-problem rank
                        correlation between score and true rate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..data.forked_tree import ForkedTree, LeanLabels, to_trajectory, trajectory_true_rate
from ..state import PolicyState
from .metrics import binary_metrics


def spearman(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    if len(a) < 2:
        return None

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else None


def _pick_metrics(scores: Sequence[float], rates: Sequence[float], correct: Sequence[Optional[float]]) -> Dict[str, Any]:
    best = max(range(len(scores)), key=lambda i: scores[i])
    known = [float(c) for c in correct if c is not None]
    return {
        "pick": rates[best],
        "random": sum(rates) / len(rates),
        "oracle": max(rates),
        "pick_correct": 0.0 if correct[best] is None else float(correct[best]),
        # answer-correctness references for the same problem: the expected
        # accuracy of a random pick and of an oracle pick (any correct trajectory)
        "random_correct": sum(known) / len(known) if known else 0.0,
        "oracle_correct": max(known) if known else 0.0,
        "rho": spearman(scores, rates),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {"n_problems": 0.0}
    n = len(rows)
    rhos = [r["rho"] for r in rows if r["rho"] is not None]
    return {
        "n_problems": float(n),
        "pick_true_rate": sum(r["pick"] for r in rows) / n,
        "random_true_rate": sum(r["random"] for r in rows) / n,
        "oracle_true_rate": sum(r["oracle"] for r in rows) / n,
        "gain_vs_random": sum(r["pick"] - r["random"] for r in rows) / n,
        "pick_correct_rate": sum(r["pick_correct"] for r in rows) / n,
        "random_correct_rate": sum(r["random_correct"] for r in rows) / n,
        "oracle_correct_rate": sum(r["oracle_correct"] for r in rows) / n,
        "within_problem_rho": sum(rhos) / len(rhos) if rhos else 0.0,
    }


def evaluate_policy_on_trees(policy: Any, trees: Sequence[ForkedTree], labels: LeanLabels) -> Dict[str, Any]:
    """`policy` is a `ProcessRewardPolicy` (needs `predict_step_probability`
    and `score_final_answer`)."""
    step_probs: List[float] = []
    step_labels: List[float] = []
    step_cache: Dict[str, float] = {}
    prm_rows: List[Dict[str, Any]] = []
    orm_rows: List[Dict[str, Any]] = []
    pair_hits, pair_total = 0, 0

    for tree in trees:
        for key, (history, step) in tree.unique_steps().items():
            if key not in labels:
                continue
            prob = policy.predict_step_probability(PolicyState(tree.question, list(history), step))
            step_cache[key] = prob
            step_probs.append(prob)
            step_labels.append(1.0 if labels[key]["proved"] else 0.0)

        for pair in tree.sibling_pairs():
            a, b = labels.get(pair.a_key), labels.get(pair.b_key)
            if a is None or b is None or bool(a["proved"]) == bool(b["proved"]):
                continue
            pa, pb = step_cache[pair.a_key], step_cache[pair.b_key]
            if pa == pb:
                pair_hits += 0.5
            elif (pa > pb) == bool(a["proved"]):
                pair_hits += 1
            pair_total += 1

        prm_scores, orm_scores, rates, correct = [], [], [], []
        for index, traj in enumerate(tree.trajectories):
            rate = trajectory_true_rate(tree, traj, labels)
            if rate is None or not traj.steps:
                continue
            keys = tree.step_keys(traj)
            prm_scores.append(sum(step_cache[k] for k in keys) / len(keys))
            orm_scores.append(policy.score_final_answer(to_trajectory(tree, index, labels)))
            rates.append(rate)
            correct.append(traj.correct)
        if len(rates) >= 2:
            prm_rows.append(_pick_metrics(prm_scores, rates, correct))
            orm_rows.append(_pick_metrics(orm_scores, rates, correct))

    return {
        "step": binary_metrics(step_probs, step_labels),
        "sibling_accuracy": pair_hits / pair_total if pair_total else None,
        "sibling_pairs": pair_total,
        "prm_pick": _aggregate(prm_rows),
        "orm_pick": _aggregate(orm_rows),
    }


def format_tree_report(rows: Dict[str, Dict[str, Any]]) -> str:
    """Two blocks: formalization (true rate of the picked trajectory) and
    answer accuracy (correctness of the picked trajectory), each with the
    random-pick and oracle-pick references from the same held-out problems."""
    nan = float("nan")
    header = f"{'policy':34s} {'stepAUC':>8s} {'sibAcc':>7s} {'PRM pick':>9s} {'ORM pick':>9s} {'random':>7s} {'oracle':>7s} {'PRM rho':>8s} {'ORM rho':>8s}"
    lines = ["formalization: true (Lean-proved) step rate of the picked trajectory", header, "-" * len(header)]
    for name, r in rows.items():
        sib = r["sibling_accuracy"]
        lines.append(
            f"{name:34s} {r['step']['auc']:8.3f} {(sib if sib is not None else nan):7.3f} "
            f"{r['prm_pick'].get('pick_true_rate', nan):9.3f} {r['orm_pick'].get('pick_true_rate', nan):9.3f} "
            f"{r['prm_pick'].get('random_true_rate', nan):7.3f} {r['prm_pick'].get('oracle_true_rate', nan):7.3f} "
            f"{r['prm_pick'].get('within_problem_rho', nan):+8.3f} {r['orm_pick'].get('within_problem_rho', nan):+8.3f}"
        )
    header2 = f"{'policy':34s} {'PRM acc':>8s} {'ORM acc':>8s} {'random':>7s} {'oracle':>7s}"
    lines += ["", "answer accuracy: correctness of the picked trajectory", header2, "-" * len(header2)]
    for name, r in rows.items():
        lines.append(
            f"{name:34s} {r['prm_pick'].get('pick_correct_rate', nan):8.3f} {r['orm_pick'].get('pick_correct_rate', nan):8.3f} "
            f"{r['prm_pick'].get('random_correct_rate', nan):7.3f} {r['prm_pick'].get('oracle_correct_rate', nan):7.3f}"
        )
    return "\n".join(lines)
