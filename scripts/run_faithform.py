#!/usr/bin/env python3
"""Policy fitting on FaithformBench, evaluation on MATH-500.

    stage fit   : train one process reward model per label setting
                  (each autoformalizer alone, and majority-vote ensembles),
                  20% validation / 80% held-out, split by problem.
    stage eval  : decode MATH-500 with the chosen policy against
                  base_greedy / random_of_k / self_consistency.
    stage report: answer accuracy, side by side.

No rule-based formalizer is involved: every step label comes from a real
autoformalizer plus DeepSeek-Prover-v2, as recorded in FaithformBench.
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

import pandas as pd

from rllf.data.dataset import IGNORE, build_pairs, build_step_examples
from rllf.data.faithform import (
    AF_MODELS,
    ALL_MODELS,
    LLM_MODELS,
    LabelPolicy,
    build_labels,
    contaminated_problem_ids,
    load_steps,
    math500_problems,
    processbench_problem_text,
    to_examples,
)
from rllf.data.trajectory import load_trajectories, save_trajectories, split_by_question
from rllf.evaluation.metrics import format_report, trajectory_report
from rllf.generation.guided_decoder import (
    MODE_PLAIN,
    MODE_RERANK,
    VERIFY_NONE,
    DecodeConfig,
    GuidedDecoder,
    self_consistency,
)
from rllf.policy.model import ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import load_policy, save_policy
from rllf.policy.training import (
    TrainConfig,
    evaluate_reward_model,
    fit_calibrators,
    train_supervised,
)
from rllf.verification.outcome_checker import default_outcome_checker


def label_settings(mode: str) -> Dict[str, LabelPolicy]:
    settings = {m: LabelPolicy(models=(m,), rule="single", name=m, mode=mode) for m in ALL_MODELS}
    settings["af_majority"] = LabelPolicy(models=AF_MODELS, rule="majority", name="af_majority", mode=mode)
    settings["llm_majority"] = LabelPolicy(models=LLM_MODELS, rule="majority", name="llm_majority", mode=mode)
    settings["all8_majority"] = LabelPolicy(models=ALL_MODELS, rule="majority", name="all8_majority", mode=mode)
    return settings


def split_examples(examples, eval_fraction: float, seed: int):
    """Split by problem so no problem appears on both sides."""
    import random

    problems = sorted({e.metadata["problem_id"] for e in examples})
    rng = random.Random(seed)
    rng.shuffle(problems)
    n_eval = int(round(len(problems) * eval_fraction))
    held = set(problems[:n_eval])
    fit = [e for e in examples if e.metadata["problem_id"] not in held]
    out = [e for e in examples if e.metadata["problem_id"] in held]
    return fit, out


def prepare(args: argparse.Namespace):
    """Steps table + problem text + the contaminated ids to hold out."""
    cache = Path(args.out) / "faithform_steps_v2.pkl"
    if cache.exists():
        table = pd.read_pickle(cache)
    else:
        table = load_steps(args.data_root, include_perturbed=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_pickle(cache)
    text = processbench_problem_text()
    contaminated = contaminated_problem_ids(text)
    return table, text, contaminated


def stage_fit(args: argparse.Namespace) -> None:
    table, text, contaminated = prepare(args)
    in_table = contaminated & set(table.problem_id.unique())
    print(f"mode = {args.mode}")
    print(f"{len(table):,} step variants over {table.problem_id.nunique()} problems")
    print(f"excluding {len(in_table)} problems whose text also appears in MATH-500\n")

    settings = label_settings(args.mode)
    chosen = args.settings or list(settings)
    rows: Dict[str, Dict[str, float]] = {}

    for name in chosen:
        policy_spec = settings[name]
        examples = to_examples(build_labels(table, policy_spec), text, exclude_problems=contaminated)
        # 20% fits the policy / 80% held out, split by problem.
        fit_ex, held_ex = split_examples(examples, eval_fraction=0.80, seed=args.seed)
        if not fit_ex or not held_ex:
            print(f"--- {name}: no usable labels, skipped"); continue
        pos = sum(e.step_reward for e in fit_ex) / max(1, len(fit_ex))
        n_pert = sum(1 for e in fit_ex if e.metadata.get("is_perturbed"))
        print(f"--- {name}: fit {len(fit_ex):,} states (pos {pos:.3f}, corrupted {n_pert:,}), "
              f"held out {len(held_ex):,}", flush=True)

        model = ProcessRewardModel(
            ProcessRewardConfig(encoder_name=args.encoder, max_length=args.max_length, pooling="cls")
        )
        config = TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            outcome_weight=0.0,  # FaithformBench traces are all correct: no outcome signal
            rank_weight=args.rank_weight, max_length=args.max_length,
            early_stopping_patience=args.patience, seed=args.seed, device=args.device,
        )
        # Early stopping only needs a stable estimate, and the held-out split is
        # 4x the size of the training split -- scoring all of it every epoch cost
        # more than the training itself.  Full evaluation happens once, after.
        import random as _random
        monitor = held_ex
        if len(held_ex) > args.monitor_size:
            monitor = _random.Random(args.seed).sample(held_ex, args.monitor_size)
        started = time.perf_counter()
        train_supervised(model, fit_ex, monitor, config, pairs=build_pairs(fit_ex),
                         on_epoch_end=lambda e, r: print(
                             f"    epoch {e}: loss={r['train_loss']:.4f}"
                             + (f" step_auc={r['step']['auc']:.3f}" if "step" in r else ""), flush=True))
        metrics = evaluate_reward_model(model, held_ex, config)["step"]
        step_scaler, outcome_scaler = fit_calibrators(model, held_ex, config)
        save_policy(str(Path(args.out) / f"prm_{args.mode}_{name}"), model, step_scaler, outcome_scaler,
                    extra={"setting": name, "mode": args.mode, "metrics": metrics})
        rows[name] = {**metrics, "fit_states": len(fit_ex), "minutes": (time.perf_counter() - started) / 60}
        print(f"    -> AUC {metrics['auc']:.3f}  acc {metrics['accuracy']:.3f}  "
              f"ECE {metrics['ece']:.3f}  ({rows[name]['minutes']:.1f} min)\n", flush=True)
        (Path(args.out) / f"fit_report_{args.mode}.json").write_text(json.dumps(rows, indent=2))

    print(f"\n=== label setting comparison, mode={args.mode} (held-out 80%) ===")
    print(format_report(rows, ("n", "auc", "accuracy", "ece", "positive_rate", "fit_states", "minutes")))


def stage_eval(args: argparse.Namespace) -> None:
    import torch
    from rllf.generation.base_model import GenerationConfig, HFBaseModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    problems = math500_problems()
    if args.sample and args.sample < len(problems):
        # Stratify by difficulty level so a subsample is not accidentally easy.
        import random as _r
        from collections import defaultdict
        buckets = defaultdict(list)
        for prob in problems:
            buckets[prob.metadata.get("level", "?")].append(prob)
        rng = _r.Random(args.seed)
        picked = []
        for level in sorted(buckets, key=str):
            group = sorted(buckets[level], key=lambda p: p.metadata.get("unique_id", p.question))
            rng.shuffle(group)
            share = round(args.sample * len(buckets[level]) / len(problems))
            picked.extend(group[:share])
        problems = picked[: args.sample]
    print(f"MATH-500: evaluating {len(problems)} problems")

    base_model = HFBaseModel(
        args.base_model, dtype=torch.bfloat16,
        generation=GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=0.95),
    )
    checker = default_outcome_checker()
    policies = {
        mode: load_policy(str(Path(args.out) / f"prm_{mode}_{args.policy_setting}"), outcome_weight=0.0)
        for mode in args.policy_modes
    }
    policy = policies[args.policy_modes[0]]

    common = dict(max_steps=args.max_steps, verification=VERIFY_NONE, seed=args.seed,
                  temperature=args.temperature)
    configs = {
        "base_greedy": DecodeConfig(mode=MODE_PLAIN, greedy=True, alpha=0.0, num_candidates=1, **common),
        "random_of_k": DecodeConfig(mode=MODE_RERANK, greedy=False, alpha=0.0, num_candidates=args.num_candidates, **common),
        "self_consistency": DecodeConfig(mode=MODE_PLAIN, greedy=False, alpha=0.0, num_candidates=1, **common),
    }
    for mode in args.policy_modes:
        configs[f"policy_guided_{mode}"] = DecodeConfig(
            mode=MODE_RERANK, greedy=True, alpha=args.alpha, num_candidates=args.num_candidates, **common
        )
    for name in (args.methods or list(configs)):
        output = Path(args.out) / f"math500_{name}.jsonl"
        done = load_trajectories(output) if output.exists() and not args.restart else []
        answered = {t.question for t in done}
        remaining = [p for p in problems if p.question not in answered]
        if not remaining:
            print(f"{name}: already complete ({len(done)})", flush=True)
            continue
        active = policies.get(name.replace("policy_guided_", ""), policy)
        decoder = GuidedDecoder(base_model, active, None, checker, configs[name])
        trajectories = list(done)
        started = time.perf_counter()
        for i, problem in enumerate(remaining):
            if name == "self_consistency":
                result = self_consistency(problem.question, problem.reference, base_model,
                                          n=args.num_candidates, config=configs[name], outcome_checker=checker)
            else:
                result = decoder.generate(problem.question, problem.reference)
            result.trajectory.metadata.update(problem.metadata)
            trajectories.append(result.trajectory)
            if (i + 1) % 10 == 0 or i + 1 == len(remaining):
                save_trajectories(output, trajectories)
                el = time.perf_counter() - started
                acc = sum(t.outcome_reward for t in trajectories) / len(trajectories)
                print(f"  {name} [{len(done)+i+1}/{len(problems)}] acc={acc:.3f} {el/(i+1):.1f}s/problem "
                      f"eta {el/(i+1)*(len(remaining)-i-1)/60:.0f}m", flush=True)
        save_trajectories(output, trajectories)
    stage_report(args)


def stage_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    rows, per = {}, {}
    for path in sorted(out.glob("math500_*.jsonl")):
        name = path.stem[len("math500_") :]
        t = load_trajectories(path)
        per[name], rows[name] = t, trajectory_report(t)
    if not rows:
        print("no MATH-500 results yet")
        return
    order = [n for n in ("base_greedy", "random_of_k", "self_consistency",
                         "policy_guided_verifiability", "policy_guided_agreement") if n in rows]
    order += [n for n in rows if n not in order]
    print("\n=== MATH-500: answer accuracy ===")
    print(format_report({n: rows[n] for n in order},
                        ("outcome_accuracy", "avg_steps", "avg_generated_tokens", "avg_latency_s")))
    base = "base_greedy" if "base_greedy" in rows else order[0]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_processbench import paired_test
    print()
    for name in order:
        if name != base:
            print(" ", paired_test(per[base], per[name], base, name))
    (out / "math500_report.json").write_text(json.dumps(rows, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["fit", "eval", "report"], default="fit")
    p.add_argument("--out", default="runs/ff")
    p.add_argument("--data-root", default="~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settings", nargs="+")
    p.add_argument("--mode", choices=["verifiability", "agreement"], default="verifiability",
                   help="verifiability = setting A (unperturbed only); agreement = setting B "
                        "(keep only verdicts matching the known ground truth)")
    p.add_argument("--encoder", default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--rank-weight", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--monitor-size", type=int, default=1500,
                   help="held-out states scored each epoch for early stopping")
    p.add_argument("--device", default="cpu")
    p.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--policy-setting", default="all8_majority")
    p.add_argument("--max-new-tokens", type=int, default=72)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-steps", type=int, default=6)
    p.add_argument("--num-candidates", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--max-test", type=int)
    p.add_argument("--sample", type=int, help="stratified subsample of MATH-500 to evaluate")
    p.add_argument("--policy-modes", nargs="+", default=["verifiability"],
                   help="which fitted policies to run as guided-decoding methods")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--restart", action="store_true")
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"fit": stage_fit, "eval": stage_eval, "report": stage_report}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
