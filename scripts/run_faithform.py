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
    self_consistency_weighted,
)
from rllf.policy.model import ProcessRewardConfig, ProcessRewardModel
from rllf.policy.scoring import load_policy, save_policy, step_formalizability_probabilities
from rllf.policy.training import (
    TrainConfig,
    evaluate_reward_model,
    fit_beam_calibration,
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
            ProcessRewardConfig(encoder_name=args.encoder, max_length=args.max_length, pooling=args.pooling)
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
    # `load_policy`/`ProcessRewardModel.load` default to CPU. That's been
    # unnoticed so far because MiniLM/ModernBERT (33-150M) score fine on CPU;
    # a 1.7B decoder-backbone policy would not, and would silently be ~50x
    # slower rather than error out. Move every loaded policy to the same
    # device the base model resolved to.
    for _policy in policies.values():
        _policy.model.to(base_model.device)
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
        # Same samples as `self_consistency`, but each one's vote is the
        # policy's reward instead of a flat 1 -- see `self_consistency_weighted`.
        configs[f"self_consistency_weighted_{mode}"] = DecodeConfig(
            mode=MODE_PLAIN, greedy=False, alpha=0.0, num_candidates=1, **common
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
            elif name.startswith("self_consistency_weighted_"):
                weight_mode = name[len("self_consistency_weighted_"):]
                result = self_consistency_weighted(
                    problem.question, problem.reference, base_model, policies[weight_mode],
                    n=args.num_candidates, config=configs[name], outcome_checker=checker,
                )
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
                         "self_consistency_weighted_verifiability", "self_consistency_weighted_agreement",
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


def stage_formalizability(args: argparse.Namespace) -> None:
    """Score already-generated MATH-500 trajectories with a fitted policy.

    Independent of answer accuracy: does a decoding method produce steps the
    trained verifiability/agreement policy scores as more formalizable? If a
    method matches `base_greedy` on accuracy but scores noticeably higher
    here, that is a free win -- the same answers, more verifiable reasoning.
    Needs no new generation; reads the `math500_*.jsonl` files `--stage eval`
    already wrote under `--out`.
    """
    out = Path(args.out)
    policy = load_policy(str(out / f"prm_{args.mode}_{args.policy_setting}"), outcome_weight=0.0)
    # Same CPU-default gap as `stage_eval` (see the comment there): fine for
    # MiniLM/ModernBERT, unusably slow -- and unbatched here, one step per
    # forward pass -- for a 1.7B-parameter policy.
    from rllf.generation.base_model import resolve_device
    policy.model.to(resolve_device())
    rows: Dict[str, Dict[str, float]] = {}
    for path in sorted(out.glob("math500_*.jsonl")):
        name = path.stem[len("math500_") :]
        trajectories = load_trajectories(path)
        step_probs: List[float] = []
        traj_means: List[float] = []
        fully = 0
        for t in trajectories:
            probs = step_formalizability_probabilities(policy, t)
            if not probs:
                continue
            step_probs.extend(probs)
            mean_p = sum(probs) / len(probs)
            traj_means.append(mean_p)
            fully += all(p > 0.5 for p in probs)
        if not traj_means:
            continue
        rows[name] = {
            "n": len(traj_means),
            "mean_step_formalizability": sum(step_probs) / len(step_probs),
            "step_formalizable_rate": sum(p > 0.5 for p in step_probs) / len(step_probs),
            "fully_formalizable_rate": fully / len(traj_means),
            "mean_trajectory_formalizability": sum(traj_means) / len(traj_means),
        }
    if not rows:
        print("no MATH-500 results to score yet")
        return
    print(f"\n=== MATH-500: formalizability under prm_{args.mode}_{args.policy_setting} ===")
    print(format_report(rows, ("n", "mean_step_formalizability", "step_formalizable_rate",
                                "fully_formalizable_rate", "mean_trajectory_formalizability")))
    (out / f"formalizability_report_{args.mode}.json").write_text(json.dumps(rows, indent=2))


def stage_fit_beam_calibration(args: argparse.Namespace) -> None:
    """2a of the `lean_constrained_beam_search` request: fit `tau_phi`
    (`ProcessRewardPolicy.beam_calibrator`, `rllf/policy/scoring.py`) for the
    classifier intervention in `rllf.generation.beam_search`.

    Reads an *already-fit* checkpoint from `--stage fit` (`--out`/`--mode`/
    `--policy-setting` locate it, same as `--stage formalizability`) and adds
    a "beam" key to its `calibration.json` -- it does not retrain or move the
    reward model itself.

    Uses a calibration split genuinely held out from what `stage_fit` used
    for `step_calibrator`/`outcome_calibrator`: those are fit directly on the
    *whole* 80% held-out split (`stage_fit`'s own eval-reporting data --
    adequate for a classification calibrator, but this is a different
    calibration target). This stage instead splits that same held-out split
    in half again (`--seed + 1`, so it's deterministic but distinct from the
    original fit/held split): one half fits `tau_phi`, the other is held out
    purely to report how the *scaled* classifier still separates
    provable/not on data neither `tau_phi` nor the original model training
    ever saw.
    """
    table, text, contaminated = prepare(args)
    settings = label_settings(args.mode)
    setting_name = (args.settings or [args.policy_setting])[0]
    policy_spec = settings[setting_name]
    examples = to_examples(build_labels(table, policy_spec), text, exclude_problems=contaminated)
    _, held_ex = split_examples(examples, eval_fraction=0.80, seed=args.seed)
    calib_ex, report_ex = split_examples(held_ex, eval_fraction=0.50, seed=args.seed + 1)
    print(f"beam calibration for prm_{args.mode}_{setting_name}: "
          f"{len(calib_ex)} states to fit tau_phi, {len(report_ex)} held out to report on", flush=True)

    checkpoint_dir = Path(args.out) / f"prm_{args.mode}_{setting_name}"
    model = ProcessRewardModel.load(checkpoint_dir)
    train_config = TrainConfig(batch_size=args.batch_size, device=args.device)
    beam_scaler = fit_beam_calibration(model, calib_ex, train_config)

    # A temperature+bias scaling is monotonic, so it cannot change this
    # split's AUC -- printed only as a sanity check that the report split
    # still discriminates at all (i.e. the checkpoint and split are sound),
    # not as evidence about tau_phi's quality specifically.
    report_metrics = evaluate_reward_model(model, report_ex, train_config)["step"]
    print(f"  tau_phi={beam_scaler.temperature:.3f} bias={beam_scaler.bias:.3f} "
          f"(report-split AUC {report_metrics['auc']:.3f}, n={report_metrics['n']:.0f})", flush=True)

    calibration_path = checkpoint_dir / "calibration.json"
    payload = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    payload["beam"] = beam_scaler.to_dict()
    calibration_path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote beam calibration into {calibration_path}/calibration.json", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--stage",
        choices=["fit", "eval", "report", "formalizability", "fit-beam-calibration"],
        default="fit",
    )
    p.add_argument("--out", default="runs/ff")
    p.add_argument("--data-root", default="~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--settings", nargs="+")
    p.add_argument("--mode", choices=["verifiability", "agreement"], default="verifiability",
                   help="verifiability = setting A (unperturbed only); agreement = setting B "
                        "(keep only verdicts matching the known ground truth)")
    p.add_argument("--encoder", default="bert-base-uncased")
    p.add_argument("--pooling", default="cls", choices=["cls", "mean", "last"],
                   help="'cls' (position 0) is meaningless for a causal decoder backbone "
                        "(nothing after it is visible under causal masking) -- use 'last' there")
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
    p.add_argument("--max-steps", type=int, default=None,
                   help="None = no artificial cap (bounded only by "
                        "MAX_STEPS_SAFETY_CEILING and the model reaching a final "
                        "answer). The old fixed default (6) zeroed out 19-52%% of "
                        "trajectories per arm with an empty final answer, at a "
                        "rate that differed systematically by arm -- pass an "
                        "explicit value to reproduce the old fixed-budget behaviour.")
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
    {"fit": stage_fit, "eval": stage_eval, "report": stage_report,
     "formalizability": stage_formalizability,
     "fit-beam-calibration": stage_fit_beam_calibration}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
