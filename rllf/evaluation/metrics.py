"""Metrics for the evaluation protocol (section 13)."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence

from ..data.trajectory import Trajectory
from ..verification.base import FORMALIZATION_FAILED

# --------------------------------------------------------------------------
# Trajectory-level metrics
# --------------------------------------------------------------------------


def step_verification_rate(trajectories: Sequence[Trajectory]) -> float:
    """sum_i sum_t r_{i,t} / sum_i T_i."""
    total = sum(len(t.steps) for t in trajectories)
    if not total:
        return 0.0
    return sum(sum(t.step_rewards) for t in trajectories) / total


def fully_verified_rate(trajectories: Sequence[Trajectory]) -> float:
    """Fraction of trajectories whose every step verified."""
    if not trajectories:
        return 0.0
    return sum(1 for t in trajectories if t.fully_verified) / len(trajectories)


def outcome_accuracy(trajectories: Sequence[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(t.outcome_reward for t in trajectories) / len(trajectories)


def formalization_success_rate(trajectories: Sequence[Trajectory]) -> float:
    """Fraction of steps the formalizer could turn into a Lean statement.

    Distinct from the verification rate: a step can formalize and then fail to
    check, which is a different failure mode from one no formalizer can express.
    """
    steps = [s for t in trajectories for s in t.steps]
    if not steps:
        return 0.0
    return sum(1 for s in steps if s.status != FORMALIZATION_FAILED) / len(steps)


def average_steps(trajectories: Sequence[Trajectory]) -> float:
    if not trajectories:
        return 0.0
    return sum(len(t.steps) for t in trajectories) / len(trajectories)


def _mean_metadata(trajectories: Sequence[Trajectory], key: str) -> float:
    values = [t.metadata.get(key) for t in trajectories if t.metadata.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def trajectory_report(trajectories: Sequence[Trajectory]) -> Dict[str, float]:
    """The full row of the evaluation table for one decoding method."""
    return {
        "n": float(len(trajectories)),
        "outcome_accuracy": outcome_accuracy(trajectories),
        "step_verification_rate": step_verification_rate(trajectories),
        "fully_verified_rate": fully_verified_rate(trajectories),
        "formalization_success_rate": formalization_success_rate(trajectories),
        "avg_steps": average_steps(trajectories),
        "avg_generated_tokens": _mean_metadata(trajectories, "generated_tokens"),
        "avg_lean_calls": _mean_metadata(trajectories, "lean_calls"),
        "avg_latency_s": _mean_metadata(trajectories, "latency_s"),
        "resample_rate": _mean_metadata(trajectories, "resample_rate"),
    }


def accuracy_by_difficulty(
    trajectories: Sequence[Trajectory], key: str = "difficulty"
) -> Dict[str, float]:
    """Outcome accuracy bucketed by a metadata field."""
    buckets: Dict[str, List[float]] = {}
    for traj in trajectories:
        bucket = str(traj.metadata.get(key, "unknown"))
        buckets.setdefault(bucket, []).append(traj.outcome_reward)
    return {name: sum(values) / len(values) for name, values in sorted(buckets.items())}


# --------------------------------------------------------------------------
# Reward-model prediction metrics
# --------------------------------------------------------------------------


def roc_auc(probabilities: Sequence[float], labels: Sequence[float]) -> float:
    """Rank-based AUC with tie handling; 0.5 when a class is absent."""
    pairs = sorted(zip(probabilities, labels))
    positives = sum(1 for _, y in pairs if y > 0.5)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    ranks: List[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average_rank
        i = j + 1
    positive_rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y > 0.5)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[float], n_bins: int = 10
) -> float:
    """Standard equal-width ECE -- the calibration check of section 14."""
    if not probabilities:
        return 0.0
    total = len(probabilities)
    error = 0.0
    for b in range(n_bins):
        low, high = b / n_bins, (b + 1) / n_bins
        members = [
            (p, y)
            for p, y in zip(probabilities, labels)
            if (p > low or (b == 0 and p >= low)) and p <= high
        ]
        if not members:
            continue
        confidence = sum(p for p, _ in members) / len(members)
        accuracy = sum(y for _, y in members) / len(members)
        error += len(members) / total * abs(confidence - accuracy)
    return error


def binary_metrics(
    probabilities: Sequence[float], labels: Sequence[float], threshold: float = 0.5
) -> Dict[str, float]:
    if not probabilities:
        return {"n": 0.0, "accuracy": 0.0, "auc": 0.5, "brier": 0.0, "bce": 0.0, "ece": 0.0, "positive_rate": 0.0}
    n = len(probabilities)
    correct = sum(1 for p, y in zip(probabilities, labels) if (p >= threshold) == (y > 0.5))
    brier = sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / n
    bce = -sum(
        y * math.log(max(p, 1e-9)) + (1 - y) * math.log(max(1 - p, 1e-9))
        for p, y in zip(probabilities, labels)
    ) / n
    return {
        "n": float(n),
        "accuracy": correct / n,
        "auc": roc_auc(probabilities, labels),
        "brier": brier,
        "bce": bce,
        "ece": expected_calibration_error(probabilities, labels),
        "positive_rate": sum(labels) / n,
    }


def pairwise_ranking_accuracy(
    preferred_scores: Sequence[float], rejected_scores: Sequence[float]
) -> float:
    """Fraction of pairs the policy orders correctly."""
    if not preferred_scores:
        return 0.0
    return sum(1 for p, r in zip(preferred_scores, rejected_scores) if p > r) / len(preferred_scores)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_report(rows: Mapping[str, Mapping[str, float]], columns: Optional[Sequence[str]] = None) -> str:
    """Render `{method: {metric: value}}` as a fixed-width comparison table."""
    if not rows:
        return "(no results)"
    columns = list(columns or sorted({c for row in rows.values() for c in row}))
    name_width = max(len(str(name)) for name in rows) + 2
    widths = [max(len(c), 9) + 2 for c in columns]
    header = "method".ljust(name_width) + "".join(c.rjust(w) for c, w in zip(columns, widths))
    lines = [header, "-" * len(header)]
    for name, row in rows.items():
        cells = []
        for column, width in zip(columns, widths):
            value = row.get(column)
            cells.append(("-" if value is None else f"{value:.3f}").rjust(width))
        lines.append(str(name).ljust(name_width) + "".join(cells))
    return "\n".join(lines)
