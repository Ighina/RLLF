#!/usr/bin/env python3
"""How far does auto-formalization reach on real ProcessBench solutions?

Runs the step verifier over ProcessBench's own *annotated correct* solutions,
per subset.  No language model is involved: this measures the reach of the
step-level reward itself -- what fraction of real reasoning steps the
formalizer can express, and how many of those Lean then confirms.

A step of a solution ProcessBench marks correct *should* verify.  Anything that
does not is either outside the formalizer's fragment (most cases) or a genuine
mismatch worth inspecting.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllf.data.processbench import SUBSETS
from rllf.verification.lean_verifier import LeanConfig, LeanStepVerifier, VerificationCache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    parser.add_argument("--per-subset", type=int, default=40, help="solutions sampled per subset")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="runs/formalization_reach.json")
    parser.add_argument("--cache", default="runs/pb/lean_cache.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset("Qwen/ProcessBench")
    verifier = LeanStepVerifier(
        config=LeanConfig(timeout=60, max_workers=args.workers),
        cache=VerificationCache(args.cache),
    )

    report = {}
    for subset in args.subsets:
        rows = [r for r in dataset[subset] if r["final_answer_correct"]][: args.per_subset]
        statuses, categories = Counter(), Counter()
        n_steps = 0
        for row in rows:
            # Each step must be verified against the steps that preceded it:
            # the formalizer binds free variables from the history, so passing
            # an empty history understates coverage and inflates
            # `unsupported_claim`.  This is sequential by construction.
            history: list = []
            for step in row["steps"]:
                result = verifier.verify(row["problem"], history, step)
                history.append(step)
                statuses[result.status] += 1
                categories[result.diagnostics.get("category", "?")] += 1
                n_steps += 1
        formalized = n_steps - statuses["formalization_failed"]
        report[subset] = {
            "solutions": len(rows),
            "steps": n_steps,
            "formalization_rate": round(formalized / max(1, n_steps), 3),
            "verification_rate": round(statuses["verified"] / max(1, n_steps), 3),
            "verified_given_formalized": round(statuses["verified"] / max(1, formalized), 3),
            "statuses": dict(statuses),
            "categories": dict(categories),
        }
        print(f"{subset}: {json.dumps(report[subset])}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
