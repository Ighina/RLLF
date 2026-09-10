#!/usr/bin/env python3
"""The ProcessBench experiment.

    stage data  : sample solutions from the frozen base model on the VALIDATION
                  split, verify every step in Lean, check every final answer,
                  and write the dataset D = {(q, tau, r_step, r_outcome)}.
    stage train : fit the process reward model on D.
    stage eval  : run the base model and the policy-guided decoder over the
                  held-out TEST split.
    stage report: answer accuracy and Lean verification rate, side by side.

    python scripts/run_processbench.py --stage all --out runs/pb

Every stage writes its artefacts, so stages can be re-run independently, and
the Lean cache is shared across all of them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllf.data.dataset import build_listwise_groups, build_pairs, build_step_examples
from rllf.data.processbench import load_processbench
from rllf.data.trajectory import Trajectory, load_trajectories, save_trajectories, split_by_question
from rllf.evaluation.metrics import format_report, trajectory_report
from rllf.generation.guided_decoder import (
    MODE_PLAIN,
    MODE_REJECTION,
    MODE_RERANK,
    VERIFY_DEFERRED,
    DecodeConfig,
    GuidedDecoder,
)
from rllf.pipeline import GeneratorSpec, Problem, TrajectoryCollector, dataset_summary
from rllf.policy.model import ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import load_policy, save_policy
from rllf.policy.training import (
    TrainConfig,
    evaluate_reward_model,
    fit_calibrators,
    train_listwise,
    train_supervised,
)
from rllf.verification.lean_verifier import LeanConfig, LeanStepVerifier, VerificationCache
from rllf.verification.outcome_checker import default_outcome_checker

REPORT_COLUMNS = (
    "outcome_accuracy",
    "step_verification_rate",
    "fully_verified_rate",
    "formalization_success_rate",
    "avg_steps",
    "avg_lean_calls",
    "avg_latency_s",
)


# --------------------------------------------------------------------------
# Shared components
# --------------------------------------------------------------------------


def build_base_model(args: argparse.Namespace):
    import torch

    from rllf.generation.base_model import GenerationConfig, HFBaseModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1" if args.offline else "0")
    return HFBaseModel(
        args.base_model,
        dtype=torch.bfloat16,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.95,
        ),
    )


def build_verifier(args: argparse.Namespace) -> LeanStepVerifier:
    return LeanStepVerifier(
        config=LeanConfig(
            project_dir=args.lean_project,
            header=args.lean_header,
            timeout=args.lean_timeout,
            max_workers=args.lean_workers,
        ),
        cache=VerificationCache(Path(args.out) / "lean_cache.jsonl"),
    )


def load_split(args: argparse.Namespace):
    path = Path(args.out) / "split.json"
    if path.exists():
        payload = json.loads(path.read_text())
        rebuild = lambda rows: [Problem(r["question"], r["reference"], r["metadata"]) for r in rows]
        return rebuild(payload["validation"]), rebuild(payload["test"])
    split = load_processbench(
        subsets=args.subsets, val_fraction=args.val_fraction, seed=args.seed, limit=args.limit
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "validation": [p.__dict__ for p in split.validation],
                "test": [p.__dict__ for p in split.test],
                "summary": split.summary(),
            },
            indent=2,
        )
    )
    print(json.dumps(split.summary(), indent=2))
    return split.validation, split.test


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_data(args: argparse.Namespace) -> None:
    validation, _ = load_split(args)
    if args.max_val:
        validation = validation[: args.max_val]
    base_model = build_base_model(args)
    verifier = build_verifier(args)

    # One generator per temperature: the policy has to see a spread of
    # reasoning styles and failure modes, not one decoding configuration.
    generators = [
        GeneratorSpec(
            name=f"{args.base_model}@T{temperature}",
            model=base_model,
            num_samples=args.samples_per_problem,
            temperature=temperature,
            max_steps=args.max_steps,
            seed=args.seed + 1000 * index,
        )
        for index, temperature in enumerate(args.temperatures)
    ]
    collector = TrajectoryCollector(generators, verifier, default_outcome_checker(), max_workers=args.lean_workers)

    output = Path(args.out) / "val_trajectories.jsonl"
    trajectories: List[Trajectory] = []
    started = time.perf_counter()
    for index, problem in enumerate(validation):
        sampled = collector.sample(problem)
        labelled = collector._annotate_all(sampled, problem.reference)
        for traj in labelled:
            traj.metadata.update({k: v for k, v in problem.metadata.items() if k != "reference_steps"})
        trajectories.extend(labelled)
        save_trajectories(output, trajectories)  # checkpoint every problem
        elapsed = time.perf_counter() - started
        rate = elapsed / (index + 1)
        print(
            f"[{index + 1}/{len(validation)}] {problem.metadata['processbench_id']} "
            f"steps/traj={sum(len(t.steps) for t in labelled) / max(1, len(labelled)):.1f} "
            f"outcome={sum(t.outcome_reward for t in labelled):.0f}/{len(labelled)} "
            f"| {rate:.1f}s/problem, eta {rate * (len(validation) - index - 1) / 60:.0f}m",
            flush=True,
        )
    print(json.dumps(dataset_summary(trajectories), indent=2))
    print(f"wrote {len(trajectories)} trajectories to {output}")


def stage_train(args: argparse.Namespace) -> None:
    trajectories = load_trajectories(Path(args.out) / "val_trajectories.jsonl")
    train_trajectories, dev_trajectories = split_by_question(trajectories, eval_fraction=0.25, seed=args.seed)
    train_examples = build_step_examples(train_trajectories)
    dev_examples = build_step_examples(dev_trajectories)
    print(
        f"{len(train_trajectories)}/{len(dev_trajectories)} trajectories -> "
        f"{len(train_examples)}/{len(dev_examples)} states"
    )

    model = ProcessRewardModel(
        ProcessRewardConfig(
            encoder_name=args.encoder,
            max_length=args.max_length,
            pooling="cls" if args.encoder else "mean",
        )
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outcome_weight=0.5,
        rank_weight=args.rank_weight,
        max_length=args.max_length,
        early_stopping_patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    pairs = build_pairs(train_examples)
    print(f"{len(pairs)} preference pairs")
    train_supervised(
        model,
        train_examples,
        dev_examples,
        config,
        pairs=pairs,
        on_epoch_end=lambda e, r: print(
            f"  epoch {e}: loss={r['train_loss']:.4f}"
            + (f" step_auc={r['step']['auc']:.3f} outcome_auc={r['outcome']['auc']:.3f}" if "step" in r else ""),
            flush=True,
        ),
    )
    if args.listwise_epochs:
        groups = build_listwise_groups(train_examples)
        print(f"listwise refinement over {len(groups)} candidate groups")
        train_listwise(model, groups, TrainConfig(**{**config.__dict__, "epochs": args.listwise_epochs,
                                                    "learning_rate": args.learning_rate / 3}))

    step_scaler, outcome_scaler = fit_calibrators(model, dev_examples, config)
    metrics = evaluate_reward_model(model, dev_examples, config)
    print(json.dumps(metrics, indent=2))
    save_policy(str(Path(args.out) / "prm"), model, step_scaler, outcome_scaler, extra={"metrics": metrics})
    print("saved policy")


def stage_eval(args: argparse.Namespace) -> None:
    _, test = load_split(args)
    if args.max_test:
        test = test[: args.max_test]
    base_model = build_base_model(args)
    verifier = build_verifier(args)
    checker = default_outcome_checker()
    policy = load_policy(str(Path(args.out) / "prm"), outcome_weight=args.outcome_weight)

    common = dict(
        max_steps=args.max_steps,
        verification=VERIFY_DEFERRED,
        seed=args.seed,
        temperature=args.temperature,
    )
    methods = {
        # Base model, unmodified.  One sample per step.
        "base_greedy": DecodeConfig(mode=MODE_PLAIN, greedy=True, alpha=0.0, num_candidates=1, **common),
        "base_sampling": DecodeConfig(mode=MODE_PLAIN, greedy=False, alpha=0.0, num_candidates=1, **common),
        # Compute-matched control: the same K candidates per step, chosen at
        # random.  The gap between this and `policy_guided` is what the policy
        # itself is worth, separated from the extra sampling.
        "random_of_k": DecodeConfig(mode=MODE_RERANK, greedy=False, alpha=0.0, num_candidates=args.num_candidates, **common),
        "policy_guided": DecodeConfig(mode=MODE_RERANK, greedy=True, alpha=args.alpha, num_candidates=args.num_candidates, **common),
        "policy_plus_verification": DecodeConfig(
            mode=MODE_REJECTION, greedy=True, alpha=args.alpha, num_candidates=args.num_candidates,
            verify_top_m=2, max_repairs=1, **common
        ),
    }
    selected = args.methods or list(methods)

    for name in selected:
        output = Path(args.out) / f"test_{name}.jsonl"
        # Resume: a method already evaluated on every test problem is skipped,
        # and a partial run continues where it stopped.  This job takes hours.
        done: List[Trajectory] = load_trajectories(output) if output.exists() and not args.restart else []
        answered = {t.question for t in done}
        remaining = [p for p in test if p.question not in answered]
        if not remaining:
            print(f"{name}: already complete ({len(done)} problems)", flush=True)
            continue
        if done:
            print(f"{name}: resuming with {len(done)} done, {len(remaining)} to go", flush=True)

        decoder = GuidedDecoder(base_model, policy, verifier, checker, methods[name])
        trajectories = list(done)
        started = time.perf_counter()
        for index, problem in enumerate(remaining):
            result = decoder.generate(problem.question, problem.reference)
            result.trajectory.metadata.update(
                {k: v for k, v in problem.metadata.items() if k != "reference_steps"}
            )
            trajectories.append(result.trajectory)
            if (index + 1) % 10 == 0 or index + 1 == len(remaining):
                save_trajectories(output, trajectories)  # checkpoint
                elapsed = time.perf_counter() - started
                accuracy = sum(t.outcome_reward for t in trajectories) / len(trajectories)
                print(
                    f"  {name} [{len(done) + index + 1}/{len(test)}] acc={accuracy:.3f} "
                    f"{elapsed / (index + 1):.1f}s/problem eta {elapsed / (index + 1) * (len(remaining) - index - 1) / 60:.0f}m",
                    flush=True,
                )
        save_trajectories(output, trajectories)
        print(f"{name}: {json.dumps(trajectory_report(trajectories))}", flush=True)
    stage_report(args)


def stage_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows: Dict[str, Dict[str, float]] = {}
    per_method: Dict[str, List[Trajectory]] = {}
    for path in sorted(out.glob("test_*.jsonl")):
        name = path.stem[len("test_") :]
        trajectories = load_trajectories(path)
        per_method[name] = trajectories
        rows[name] = trajectory_report(trajectories)
    if not rows:
        print("no test results yet")
        return
    order = [n for n in ("base_greedy", "base_sampling", "random_of_k", "policy_guided", "policy_plus_verification") if n in rows]
    order += [n for n in rows if n not in order]
    print("\n=== ProcessBench: base model vs policy-guided decoding ===")
    print(format_report({n: rows[n] for n in order}, REPORT_COLUMNS))

    baseline = "base_greedy" if "base_greedy" in rows else order[0]
    print(f"\ndeltas vs {baseline}:")
    for name in order:
        if name == baseline:
            continue
        d_acc = rows[name]["outcome_accuracy"] - rows[baseline]["outcome_accuracy"]
        d_ver = rows[name]["step_verification_rate"] - rows[baseline]["step_verification_rate"]
        print(f"  {name:26} answer {d_acc:+.3f}   step-verification {d_ver:+.3f}")

    if len(per_method) > 1 and "policy_guided" in per_method and baseline in per_method:
        print()
        print(paired_test(per_method[baseline], per_method["policy_guided"], baseline, "policy_guided"))
    (out / "report.json").write_text(json.dumps(rows, indent=2))


def stage_diagnose(args: argparse.Namespace) -> None:
    """Does the reward model generalise to the held-out problems?

    Re-scores the steps of the *test* trajectories, whose Lean labels were
    recorded during evaluation.  This is the diagnostic that explains the
    headline table: a policy that cannot separate verified from unverified
    steps on unseen questions cannot help the decoder, however good its
    training-split numbers were.
    """
    from rllf.evaluation.metrics import binary_metrics
    from rllf.policy.training import predict

    out = Path(args.out)
    policy = load_policy(str(out / "prm"), outcome_weight=args.outcome_weight)
    rows: Dict[str, Dict[str, float]] = {}
    for path in sorted(out.glob("test_*.jsonl")):
        name = path.stem[len("test_") :]
        trajectories = load_trajectories(path)
        examples = [e for e in build_step_examples(trajectories) if not e.is_final]
        if not examples:
            continue
        predictions = predict(policy.model, examples, batch_size=32)
        step_probs = [policy.step_calibrator.apply(__import__("torch").tensor(v)).item()
                      for v in predictions["step_logit"]]
        labels = [e.step_reward for e in examples]
        outcome_probs = predictions["outcome_prob"]
        outcome_labels = [e.outcome_reward for e in examples]
        rows[name] = {
            **{f"step_{k}": v for k, v in binary_metrics(step_probs, labels).items()},
            **{f"outcome_{k}": v for k, v in binary_metrics(outcome_probs, outcome_labels).items()},
        }
    if not rows:
        print("no test trajectories to diagnose")
        return
    print("\n=== process reward model on held-out TEST states ===")
    print(format_report(rows, ("step_n", "step_auc", "step_accuracy", "step_ece", "step_positive_rate",
                               "outcome_auc", "outcome_ece")))
    (out / "diagnose.json").write_text(json.dumps(rows, indent=2))


def paired_test(a: Sequence[Trajectory], b: Sequence[Trajectory], name_a: str, name_b: str) -> str:
    """Exact McNemar test over the questions both methods answered."""
    from math import comb

    by_question_a = {t.question: t.outcome_reward for t in a}
    by_question_b = {t.question: t.outcome_reward for t in b}
    shared = set(by_question_a) & set(by_question_b)
    b_only = sum(1 for q in shared if by_question_b[q] > by_question_a[q])
    a_only = sum(1 for q in shared if by_question_a[q] > by_question_b[q])
    n = a_only + b_only
    if n == 0:
        return f"paired on {len(shared)} questions: no disagreements"
    tail = sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / (2**n)
    return (
        f"paired on {len(shared)} questions: {name_b} wins {b_only}, {name_a} wins {a_only}, "
        f"two-sided exact McNemar p = {min(1.0, 2 * tail):.3f}"
    )


# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage", choices=["data", "train", "eval", "report", "diagnose", "all"], default="all"
    )
    parser.add_argument("--out", default="runs/pb")
    parser.add_argument("--subsets", nargs="+", default=["gsm8k"])
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, help="cap the number of problems before splitting")
    parser.add_argument("--max-val", type=int, help="cap the validation problems actually generated on")
    parser.add_argument("--max-test", type=int, help="cap the test problems actually evaluated")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    parser.add_argument("--samples-per-problem", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)

    parser.add_argument("--lean-project")
    parser.add_argument("--lean-header", default="")
    parser.add_argument("--lean-timeout", type=float, default=60.0)
    parser.add_argument("--lean-workers", type=int, default=4)

    parser.add_argument("--encoder", default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--listwise-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", help="force a torch device for policy training (e.g. cpu)")

    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--restart", action="store_true", help="ignore existing test_*.jsonl checkpoints")
    args = parser.parse_args(argv)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    stages = ["data", "train", "eval"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n{'=' * 20} stage: {stage} {'=' * 20}", flush=True)
        {
            "data": stage_data,
            "train": stage_train,
            "eval": stage_eval,
            "report": stage_report,
            "diagnose": stage_diagnose,
        }[stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
