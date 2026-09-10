#!/usr/bin/env python3
"""Command-line entry point for the RLLF pipeline.

    python train.py generate --out data/trajectories.jsonl --synthetic 60
    python train.py train    --trajectories data/trajectories.jsonl --out runs/prm
    python train.py evaluate --policy runs/prm --synthetic 24
    python train.py demo                      # generate + train + evaluate

Every subcommand runs offline by default (synthetic problems, a scripted base
model, and the rule-based formalizer).  Point `--base-model` at a Hugging Face
checkpoint and drop `--mock-verifier` to run the real thing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from rllf.data.dataset import build_listwise_groups, build_pairs, build_step_examples
from rllf.data.synthetic import SyntheticBaseModel, make_problems
from rllf.data.trajectory import load_trajectories, split_by_question
from rllf.evaluation.protocol import alpha_ablation, difficulty_table, report_table, run_evaluation
from rllf.pipeline import GeneratorSpec, Problem, TrajectoryCollector, dataset_summary, load_problems
from rllf.policy.model import ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import ProcessRewardPolicy, load_policy, save_policy
from rllf.policy.training import (
    TrainConfig,
    evaluate_reward_model,
    fit_calibrators,
    train_listwise,
    train_pairwise,
    train_supervised,
)
from rllf.verification.lean_verifier import LeanConfig, LeanStepVerifier, MockStepVerifier, VerificationCache
from rllf.verification.outcome_checker import default_outcome_checker


# --------------------------------------------------------------------------
# Component construction
# --------------------------------------------------------------------------


def build_verifier(args: argparse.Namespace):
    if args.no_verifier:
        return None
    if args.mock_verifier:
        return MockStepVerifier()
    config = LeanConfig(
        project_dir=args.lean_project,
        header=args.lean_header,
        timeout=args.lean_timeout,
        max_workers=args.lean_workers,
    )
    return LeanStepVerifier(config=config, cache=VerificationCache(args.cache))


def build_problems(args: argparse.Namespace) -> List[Problem]:
    if args.problems:
        return load_problems(args.problems)
    return make_problems(args.synthetic, seed=args.seed)


def build_base_models(args: argparse.Namespace, problems: Sequence[Problem]) -> List[GeneratorSpec]:
    """One generator per (model, temperature, style) combination."""
    if args.base_model:
        from rllf.generation.base_model import GenerationConfig, HFBaseModel

        specs = []
        for index, temperature in enumerate(args.temperatures):
            model = HFBaseModel(
                args.base_model,
                generation=GenerationConfig(temperature=temperature, max_new_tokens=args.max_new_tokens),
            )
            specs.append(
                GeneratorSpec(
                    name=f"{args.base_model}@T{temperature}",
                    model=model,
                    num_samples=args.samples_per_problem,
                    temperature=temperature,
                    max_steps=args.max_steps,
                    seed=args.seed + 100 * index,
                )
            )
        return specs
    # Offline: several synthetic generators with different error rates and
    # surface styles, so the policy sees diverse reasoning and failure modes.
    styles = ["plain", "terse", "verbose"]
    return [
        GeneratorSpec(
            name=f"synthetic-{style}-e{error:.2f}",
            model=SyntheticBaseModel(problems, error_rate=error, seed=args.seed + 7 * index, style=style),
            num_samples=args.samples_per_problem,
            temperature=0.8 + 0.2 * index,
            max_steps=args.max_steps,
            seed=args.seed + 13 * index,
        )
        for index, (style, error) in enumerate(zip(styles, [0.25, 0.4, 0.55]))
    ]


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    problems = build_problems(args)
    collector = TrajectoryCollector(
        generators=build_base_models(args, problems),
        verifier=build_verifier(args),
        outcome_checker=default_outcome_checker(),
        max_workers=args.workers,
    )
    trajectories = collector.collect(problems, output_path=args.out, progress=args.progress)
    print(json.dumps(dataset_summary(trajectories), indent=2))
    if args.out:
        print(f"\nwrote {len(trajectories)} trajectories to {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    trajectories = load_trajectories(args.trajectories)
    train_trajectories, eval_trajectories = split_by_question(
        trajectories, eval_fraction=args.eval_fraction, seed=args.seed
    )
    train_examples = build_step_examples(train_trajectories)
    eval_examples = build_step_examples(eval_trajectories)
    print(
        f"{len(train_trajectories)} train / {len(eval_trajectories)} eval trajectories "
        f"-> {len(train_examples)} / {len(eval_examples)} states"
    )

    model = ProcessRewardModel(
        ProcessRewardConfig(
            encoder_name=args.encoder,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            max_length=args.max_length,
        )
    )
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outcome_weight=args.outcome_weight,
        rank_weight=args.rank_weight,
        max_length=args.max_length,
        seed=args.seed,
        log_every=args.log_every,
    )
    pairs = build_pairs(train_examples) if args.rank_weight > 0 else None
    if pairs:
        print(f"{len(pairs)} preference pairs for the auxiliary ranking loss")

    history = train_supervised(
        model,
        train_examples,
        eval_examples if eval_examples else None,
        config,
        pairs=pairs,
        on_epoch_end=lambda epoch, record: print(
            f"epoch {epoch}: loss={record['train_loss']:.4f} "
            + (
                f"step_auc={record['step']['auc']:.3f} outcome_auc={record['outcome']['auc']:.3f}"
                if "step" in record
                else ""
            )
        ),
    )
    if args.pairwise_epochs and pairs:
        print("refining with the pairwise ranking objective")
        train_pairwise(model, pairs, config=TrainConfig(**{**config.__dict__, "epochs": args.pairwise_epochs}))
    if args.listwise_epochs:
        # Listwise ranking optimises the decision the decoder actually makes:
        # which of several continuations of *this* state to accept.
        groups = build_listwise_groups(train_examples)
        print(f"refining with the listwise objective over {len(groups)} candidate groups")
        train_listwise(
            model,
            groups,
            TrainConfig(**{**config.__dict__, "epochs": args.listwise_epochs, "learning_rate": config.learning_rate / 3}),
        )

    # Calibration is fitted on held-out states: a raw reward logit is not a
    # log-probability and must not be fed to the decoder as one.
    calibration_source = eval_examples or train_examples
    step_scaler, outcome_scaler = fit_calibrators(model, calibration_source, config)
    metrics = evaluate_reward_model(model, calibration_source, config)
    print(json.dumps(metrics, indent=2))
    print(f"calibration: step={step_scaler.to_dict()} outcome={outcome_scaler.to_dict()}")

    save_policy(args.out, model, step_scaler, outcome_scaler, extra={"metrics": metrics})
    print(f"saved policy to {args.out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    problems = build_problems(args)
    specs = build_base_models(args, problems)
    base_model = specs[0].model
    policy: Optional[ProcessRewardPolicy] = None
    if args.policy:
        policy = load_policy(args.policy, outcome_weight=args.outcome_weight)
    verifier = build_verifier(args)
    checker = default_outcome_checker()

    results = run_evaluation(
        problems,
        base_model,
        policy=policy,
        verifier=verifier,
        outcome_checker=checker,
        num_candidates=args.num_candidates,
        best_of=args.best_of,
        alpha=args.alpha,
        max_steps=args.max_steps,
        include_oracle=args.oracle,
        seed=args.seed,
    )
    print("\n=== evaluation protocol ===")
    print(report_table(results))
    print("\n=== accuracy by difficulty ===")
    print(difficulty_table(results))

    if policy is not None and args.ablation:
        ablation = alpha_ablation(
            problems, base_model, policy, verifier, checker,
            alphas=args.alphas, num_candidates=args.num_candidates, max_steps=args.max_steps, seed=args.seed,
        )
        print("\n=== guidance-strength ablation (alpha) ===")
        print(report_table(ablation))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Generate -> train -> evaluate, end to end, in one process."""
    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)
    trajectory_path = workdir / "trajectories.jsonl"
    policy_path = workdir / "prm"

    print("=== 1/3 generating and verifying trajectories ===")
    generate_args = argparse.Namespace(**{**vars(args), "out": str(trajectory_path)})
    cmd_generate(generate_args)

    print("\n=== 2/3 training the process reward model ===")
    train_args = argparse.Namespace(
        **{**vars(args), "trajectories": str(trajectory_path), "out": str(policy_path)}
    )
    cmd_train(train_args)

    print("\n=== 3/3 evaluating guided decoding ===")
    # A different seed draws different problems: evaluating on the questions
    # the reward model trained on would report memorisation as accuracy.
    eval_args = argparse.Namespace(
        **{
            **vars(args),
            "policy": str(policy_path),
            "synthetic": args.eval_problems,
            "seed": args.seed + 1000,
        }
    )
    return cmd_evaluate(eval_args)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(sub: argparse.ArgumentParser) -> None:
        # Repeated per subcommand so `train.py evaluate --seed 3` works as well
        # as `train.py --seed 3 evaluate`.
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--problems", help="JSONL file of {question, reference}")
        sub.add_argument("--synthetic", type=int, default=48, help="number of synthetic problems")
        sub.add_argument("--base-model", help="Hugging Face checkpoint (frozen)")
        sub.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
        sub.add_argument("--samples-per-problem", type=int, default=4)
        sub.add_argument("--max-steps", type=int, default=4)
        sub.add_argument("--max-new-tokens", type=int, default=128)

    def add_verifier_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--mock-verifier", action="store_true", help="evaluate claims in Python instead of Lean")
        sub.add_argument("--no-verifier", action="store_true")
        sub.add_argument("--lean-project", help="lake project directory (for mathlib)")
        sub.add_argument("--lean-header", default="", help="e.g. 'import Mathlib'")
        sub.add_argument("--lean-timeout", type=float, default=30.0)
        sub.add_argument("--lean-workers", type=int, default=4)
        sub.add_argument("--cache", default="runs/lean_cache.jsonl")
        sub.add_argument("--workers", type=int, default=4)

    def add_train_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--encoder", default=None, help="HF encoder name; default is the built-in one")
        sub.add_argument("--hidden-size", type=int, default=128)
        sub.add_argument("--layers", type=int, default=2)
        sub.add_argument("--max-length", type=int, default=384)
        sub.add_argument("--epochs", type=int, default=8)
        sub.add_argument("--pairwise-epochs", type=int, default=0)
        sub.add_argument("--listwise-epochs", type=int, default=0)
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--learning-rate", type=float, default=3e-4)
        sub.add_argument("--outcome-weight", type=float, default=0.5)
        sub.add_argument("--rank-weight", type=float, default=0.5)
        sub.add_argument("--eval-fraction", type=float, default=0.2)
        sub.add_argument("--log-every", type=int, default=0)

    def add_eval_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--num-candidates", type=int, default=4)
        sub.add_argument("--best-of", type=int, default=4)
        sub.add_argument("--alpha", type=float, default=1.0)
        sub.add_argument("--ablation", action="store_true")
        sub.add_argument(
            "--oracle", action="store_true",
            help="also run the verifier-as-policy skyline (one Lean call per candidate)",
        )
        sub.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 4.0])

    generate = subparsers.add_parser("generate", help="sample and verify trajectories")
    add_data_args(generate)
    add_verifier_args(generate)
    generate.add_argument("--out", default="runs/trajectories.jsonl")
    generate.add_argument("--progress", action="store_true")
    generate.set_defaults(func=cmd_generate)

    train = subparsers.add_parser("train", help="train the process reward model")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--trajectories", required=True)
    train.add_argument("--out", default="runs/prm")
    add_train_args(train)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="run the evaluation protocol")
    add_data_args(evaluate)
    add_verifier_args(evaluate)
    add_eval_args(evaluate)
    evaluate.add_argument("--policy", help="directory saved by `train`")
    evaluate.add_argument("--outcome-weight", type=float, default=0.5)
    evaluate.set_defaults(func=cmd_evaluate)

    demo = subparsers.add_parser("demo", help="generate + train + evaluate end to end")
    add_data_args(demo)
    add_verifier_args(demo)
    add_train_args(demo)
    add_eval_args(demo)
    demo.add_argument("--out", default="runs/demo")
    demo.add_argument("--eval-problems", type=int, default=24)
    demo.add_argument("--progress", action="store_true")
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
