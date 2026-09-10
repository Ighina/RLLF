from .metrics import (
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

# `protocol` pulls in the decoder, which imports the policy package, which
# imports these metrics -- so it is resolved lazily to keep that cycle from
# forming at import time.
_PROTOCOL_EXPORTS = frozenset(
    {"MethodResult", "alpha_ablation", "difficulty_table", "report_table", "run_evaluation", "REPORT_COLUMNS"}
)


def __getattr__(name: str):
    if name in _PROTOCOL_EXPORTS:
        from . import protocol

        return getattr(protocol, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _PROTOCOL_EXPORTS)


__all__ = [
    "step_verification_rate",
    "fully_verified_rate",
    "outcome_accuracy",
    "formalization_success_rate",
    "trajectory_report",
    "accuracy_by_difficulty",
    "binary_metrics",
    "roc_auc",
    "expected_calibration_error",
    "pairwise_ranking_accuracy",
    "format_report",
    "run_evaluation",
    "alpha_ablation",
    "report_table",
    "difficulty_table",
    "MethodResult",
]
