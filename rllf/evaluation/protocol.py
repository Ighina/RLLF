"""The evaluation protocol of section 13.

Runs the six required comparisons on one set of problems and reports final
accuracy alongside process metrics and cost:

1. base model, greedy decoding;
2. base model, temperature sampling;
3. base model, best-of-N (ranked by the reward model);
4. base model, verification-only rejection sampling;
5. base model guided by the learned policy;
6. base model guided by the policy *and* verification.

The key ablation -- guidance strength `alpha` -- is :func:`alpha_ablation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.trajectory import Trajectory
from ..generation.guided_decoder import (
    MODE_PLAIN,
    MODE_REJECTION,
    MODE_RERANK,
    VERIFY_DEFERRED,
    VERIFY_NONE,
    DecodeConfig,
    DecodeResult,
    GuidedDecoder,
    best_of_n,
)
from ..pipeline import Problem
from ..policy.scoring import PolicyAdapter, UniformPolicy, VerifierPolicy
from .metrics import accuracy_by_difficulty, format_report, trajectory_report

REPORT_COLUMNS = (
    "outcome_accuracy",
    "step_verification_rate",
    "fully_verified_rate",
    "formalization_success_rate",
    "avg_steps",
    "avg_generated_tokens",
    "avg_lean_calls",
    "avg_latency_s",
    "resample_rate",
)


@dataclass
class MethodResult:
    name: str
    trajectories: List[Trajectory]
    report: Dict[str, float] = field(default_factory=dict)


def _run(
    name: str,
    problems: Sequence[Problem],
    decode: Callable[[Problem], DecodeResult],
    verifier: Any = None,
) -> MethodResult:
    trajectories = []
    for problem in problems:
        result = decode(problem)
        # Carry the problem's own metadata so per-difficulty breakdowns work.
        result.trajectory.metadata.update(problem.metadata)
        trajectories.append(result.trajectory)
    return MethodResult(name=name, trajectories=trajectories, report=trajectory_report(trajectories))


def _decoder(
    base_model: Any,
    policy: Optional[PolicyAdapter],
    verifier: Any,
    outcome_checker: Any,
    config: DecodeConfig,
) -> Callable[[Problem], DecodeResult]:
    decoder = GuidedDecoder(base_model, policy, verifier, outcome_checker, config)
    return lambda problem: decoder.generate(problem.question, problem.reference)


def run_evaluation(
    problems: Sequence[Problem],
    base_model: Any,
    policy: Optional[PolicyAdapter] = None,
    verifier: Any = None,
    outcome_checker: Any = None,
    num_candidates: int = 4,
    best_of: int = 4,
    alpha: float = 1.0,
    max_steps: int = 8,
    methods: Optional[Sequence[str]] = None,
    include_oracle: bool = False,
    seed: int = 0,
) -> Dict[str, MethodResult]:
    """Run every baseline plus the guided variants on the same problems.

    All methods are scored with the same outcome checker, and every trajectory
    is verified (deferred) so that process metrics are comparable even for the
    baselines that never consult Lean during decoding.
    """
    base_kwargs = dict(max_steps=max_steps, num_candidates=num_candidates, seed=seed)
    scoring_policy = policy or UniformPolicy()
    # Baselines are still measured on process quality, so they verify after the
    # fact; only their `avg_lean_calls` is charged during decoding.
    audit = VERIFY_DEFERRED if verifier is not None else VERIFY_NONE

    specs: Dict[str, Callable[[], MethodResult]] = {
        "greedy": lambda: _run(
            "greedy",
            problems,
            _decoder(
                base_model, None, verifier, outcome_checker,
                DecodeConfig(mode=MODE_PLAIN, greedy=True, alpha=0.0, verification=audit, **base_kwargs),
            ),
        ),
        "sampling": lambda: _run(
            "sampling",
            problems,
            _decoder(
                base_model, None, verifier, outcome_checker,
                DecodeConfig(mode=MODE_PLAIN, greedy=False, alpha=0.0, verification=audit, **base_kwargs),
            ),
        ),
        "best_of_n": lambda: _run(
            "best_of_n",
            problems,
            lambda problem: best_of_n(
                problem.question,
                problem.reference,
                base_model,
                scoring_policy,
                n=best_of,
                config=DecodeConfig(mode=MODE_PLAIN, greedy=False, alpha=0.0, verification=audit, **base_kwargs),
                verifier=verifier,
                outcome_checker=outcome_checker,
                rank_by="policy",
            ),
        ),
        "verification_only": lambda: _run(
            "verification_only",
            problems,
            _decoder(
                base_model, UniformPolicy(), verifier, outcome_checker,
                DecodeConfig(mode=MODE_REJECTION, greedy=False, alpha=0.0, verification=audit, **base_kwargs),
            ),
        ),
        "policy_guided": lambda: _run(
            "policy_guided",
            problems,
            _decoder(
                base_model, scoring_policy, verifier, outcome_checker,
                DecodeConfig(mode=MODE_RERANK, greedy=True, alpha=alpha, verification=audit, **base_kwargs),
            ),
        ),
        "policy_plus_verification": lambda: _run(
            "policy_plus_verification",
            problems,
            _decoder(
                base_model, scoring_policy, verifier, outcome_checker,
                DecodeConfig(mode=MODE_REJECTION, greedy=True, alpha=alpha, verification=audit, **base_kwargs),
            ),
        ),
        # Skyline: what a *perfect* process reward model would buy.  Not a
        # baseline -- it calls the verifier once per candidate -- but it
        # separates "the reward model is weak" from "guided decoding is broken".
        "oracle_policy": lambda: _run(
            "oracle_policy",
            problems,
            _decoder(
                base_model, VerifierPolicy(verifier), verifier, outcome_checker,
                DecodeConfig(mode=MODE_RERANK, greedy=True, alpha=alpha, verification=audit, **base_kwargs),
            ),
        ),
    }

    default_methods = [name for name in specs if name != "oracle_policy"]
    if include_oracle:
        default_methods.append("oracle_policy")
    selected = list(methods) if methods else default_methods
    if policy is None:
        selected = [m for m in selected if m not in ("best_of_n", "policy_guided", "policy_plus_verification")]
    if verifier is None:
        selected = [
            m for m in selected
            if m not in ("verification_only", "policy_plus_verification", "oracle_policy")
        ]
    return {name: specs[name]() for name in selected if name in specs}


def alpha_ablation(
    problems: Sequence[Problem],
    base_model: Any,
    policy: PolicyAdapter,
    verifier: Any = None,
    outcome_checker: Any = None,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    mode: str = MODE_RERANK,
    num_candidates: int = 4,
    max_steps: int = 8,
    greedy: bool = False,
    seed: int = 0,
) -> Dict[str, MethodResult]:
    """Vary guidance strength; alpha = 0 is the unmodified base model."""
    results: Dict[str, MethodResult] = {}
    for alpha in alphas:
        config = DecodeConfig(
            mode=mode,
            alpha=alpha,
            greedy=greedy,
            num_candidates=num_candidates,
            max_steps=max_steps,
            verification=VERIFY_DEFERRED if verifier is not None else VERIFY_NONE,
            seed=seed,
        )
        results[f"alpha={alpha:g}"] = _run(
            f"alpha={alpha:g}",
            problems,
            _decoder(base_model, policy, verifier, outcome_checker, config),
        )
    return results


def report_table(results: Dict[str, MethodResult], columns: Sequence[str] = REPORT_COLUMNS) -> str:
    return format_report({name: r.report for name, r in results.items()}, columns)


def difficulty_table(results: Dict[str, MethodResult]) -> str:
    rows = {name: accuracy_by_difficulty(r.trajectories) for name, r in results.items()}
    columns = sorted({c for row in rows.values() for c in row})
    return format_report(rows, columns)
