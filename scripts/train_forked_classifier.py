#!/usr/bin/env python3
"""Retrain the step classifier on forked-tree data (experiment 1 of
`forked-tree-classifier-retraining.md`): several solutions per problem from
the decoded model itself, every step labelled by the real Lean pipeline.

    python3 scripts/train_forked_classifier.py \\
        --samples /scratch/users/iacopog/rllf-runs/forked_tree/samples.jsonl \\
        --labels  /scratch/users/iacopog/rllf-runs/forked_tree/real_lean.json \\
        --out     /scratch/users/iacopog/rllf-runs/forked_tree/classifiers \\
        --objective pairwise \\
        --compare runs/ff/prm_verifiability_all8_majority \\
                  /scratch/users/iacopog/rllf-runs/ff_qwen3-1.7b/prm_verifiability_all8_majority

Objectives (all on the identical by-problem 80/20 tree split, all evaluated
by `rllf.evaluation.forked_tree_eval` on the held-out trees):

  supervised  masked BCE on the step head (proved) + outcome head (correct)
              -- the old target on the new data; the control.
  pairwise    -log sigmoid(s+ - s-) over within-problem pairs only: the
              fork siblings (same prefix) plus cross-trajectory pairs of the
              same problem. Step target unchanged (proved); what changes is
              that the problem is held fixed inside every loss term.
  listwise    softmax CE over each problem's proved steps (same idea,
              list-shaped).
  orm         outcome-head listwise ranking of a problem's *whole*
              trajectories by true formalization rate (`--orm-target
              correct` ranks by answer correctness instead).

`--compare` scores existing policies on the same held-out trees so the
new fits are read against the FaithformBench-trained ones.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllf.data.dataset import build_listwise_groups, build_pairs
from rllf.data.forked_tree import (
    build_prm_examples,
    build_problem_pairs,
    build_sibling_pairs,
    build_trajectory_groups,
    load_trees,
    split_trees,
    summarize,
)
from rllf.evaluation.forked_tree_eval import evaluate_policy_on_trees, format_tree_report
from rllf.policy.model import ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import ProcessRewardPolicy, load_policy, save_policy
from rllf.policy.training import (
    TrainConfig,
    fit_calibrators,
    train_listwise,
    train_pairwise,
    train_supervised,
    train_trajectory_listwise,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--objective", default="pairwise", choices=["supervised", "pairwise", "listwise", "orm"])
    p.add_argument("--orm-target", default="true_rate", choices=["true_rate", "correct"])
    p.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"])
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--outcome-weight", type=float, default=0.5, help="supervised only: lambda on the outcome head")
    p.add_argument("--rank-weight", type=float, default=0.0, help="supervised only: auxiliary pair loss weight")
    p.add_argument("--pairs-per-problem", type=int, default=16)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=0, help="early-stopping patience in epochs (0 = train all epochs, keep best held-out epoch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--compare", nargs="*", default=[], help="existing policy dirs to score on the same held-out trees")
    p.add_argument("--name", default=None, help="checkpoint name (default: forked_<objective>)")
    args = p.parse_args(argv)

    trees = load_trees(args.samples)
    labels = json.load(open(args.labels))["detail"]
    fit_trees, held_trees = split_trees(trees, args.eval_fraction, args.seed)
    print(f"{len(trees)} trees: fit {len(fit_trees)}, held out {len(held_trees)}")
    print("fit:      ", json.dumps(summarize(fit_trees, labels)))
    print("held out: ", json.dumps(summarize(held_trees, labels)), flush=True)

    fit_examples = build_prm_examples(fit_trees, labels)
    held_examples = build_prm_examples(held_trees, labels)
    config = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        outcome_weight=args.outcome_weight, rank_weight=args.rank_weight, max_length=args.max_length,
        seed=args.seed, device=args.device, early_stopping_patience=args.patience,
    )
    model = ProcessRewardModel(
        ProcessRewardConfig(encoder_name=args.encoder, max_length=args.max_length, pooling=args.pooling)
    )
    started = time.perf_counter()
    if args.objective == "supervised":
        pairs = build_sibling_pairs(fit_trees, labels) + build_problem_pairs(fit_examples, args.pairs_per_problem, args.seed)
        print(f"supervised: {len(fit_examples)} states, {len(pairs)} aux pairs (rank_weight={args.rank_weight})", flush=True)
        history = train_supervised(model, fit_examples, held_examples, config, pairs=pairs,
                                   on_epoch_end=lambda e, r: print(f"    epoch {e}: loss={r['train_loss']:.4f}"
                                                                   + (f" held step_auc={r['step']['auc']:.3f}" if "step" in r else ""), flush=True))
    elif args.objective == "pairwise":
        sibling = build_sibling_pairs(fit_trees, labels)
        problem = build_problem_pairs(fit_examples, args.pairs_per_problem, args.seed)
        prefix = [pp for pp in build_pairs(fit_examples, allow_question_level=False) ]
        pairs = sibling + problem + prefix
        print(f"pairwise: {len(sibling)} sibling + {len(problem)} problem + {len(prefix)} other same-prefix pairs", flush=True)
        eval_pairs = build_sibling_pairs(held_trees, labels) + build_problem_pairs(held_examples, args.pairs_per_problem, args.seed)
        history = train_pairwise(model, pairs, eval_pairs, config)
        for e, r in enumerate(history["eval"]):
            print(f"    epoch {e}: loss={history['train_loss'][e]:.4f} held pair acc={r['ranking_accuracy']:.3f}", flush=True)
    elif args.objective == "listwise":
        # one list per problem (all its labelled steps) plus the same-prefix lists
        by_problem = {}
        for e in fit_examples:
            by_problem.setdefault(e.metadata["problem_id"], []).append(e)
        groups = [g for g in by_problem.values() if len({x.step_reward for x in g}) > 1]
        groups += build_listwise_groups(fit_examples)
        eval_pairs = build_sibling_pairs(held_trees, labels) + build_problem_pairs(held_examples, args.pairs_per_problem, args.seed)
        print(f"listwise: {len(groups)} lists, {len(eval_pairs)} held-out pairs", flush=True)
        history = train_listwise(model, groups, config, eval_pairs=eval_pairs)
        for e, r in enumerate(history["eval"]):
            print(f"    epoch {e}: loss={history['train_loss'][e]:.4f} held pair acc={r['ranking_accuracy']:.3f}", flush=True)
    else:
        groups = build_trajectory_groups(fit_trees, labels)
        eval_groups = build_trajectory_groups(held_trees, labels)
        print(f"orm ({args.orm_target}): {len(groups)} fit groups, {len(eval_groups)} held-out groups", flush=True)
        history = train_trajectory_listwise(model, groups, config, target=args.orm_target, eval_groups=eval_groups)
        for e, r in enumerate(history["eval"]):
            print(f"    epoch {e}: loss={history['train_loss'][e]:.4f} held pick={r['pick_true_rate']:.3f} "
                  f"(random {r['random_true_rate']:.3f}, oracle {r['oracle_true_rate']:.3f})", flush=True)
    minutes = (time.perf_counter() - started) / 60
    if "best_epoch" in history:
        print(f"    best epoch {history['best_epoch']} by {history['select_by']} = {history['best_score']:.3f}", flush=True)

    step_scaler, outcome_scaler = fit_calibrators(model, held_examples, config)
    policy = ProcessRewardPolicy(model, outcome_weight=0.0, step_calibrator=step_scaler, outcome_calibrator=outcome_scaler)
    name = args.name or f"forked_{args.objective}" + (f"_{args.orm_target}" if args.objective == "orm" else "") \
        + (f"_rank{args.rank_weight:g}" if args.objective == "supervised" and args.rank_weight else "")
    rows = {name: evaluate_policy_on_trees(policy, held_trees, labels)}
    for path in args.compare:
        other = load_policy(path, outcome_weight=0.0)
        other.model.to(next(model.parameters()).device)
        rows[Path(path).name] = evaluate_policy_on_trees(other, held_trees, labels)

    print(f"\n=== held-out trees ({len(held_trees)} problems), {minutes:.1f} min training ===")
    print(format_tree_report(rows), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_policy(str(out / name), model, step_scaler, outcome_scaler,
                extra={"objective": args.objective, "orm_target": args.orm_target, "encoder": args.encoder,
                       "seed": args.seed, "eval_fraction": args.eval_fraction, "held_out": rows[name]})
    report = {"args": vars(args), "minutes": minutes, "history": history, "held_out": rows,
              "fit_summary": summarize(fit_trees, labels), "held_summary": summarize(held_trees, labels)}
    (out / f"report_{name}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"saved {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
