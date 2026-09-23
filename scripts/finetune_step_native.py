#!/usr/bin/env python3
"""Fine-tune a base model to natively emit `<step>`/`<final>` between
reasoning steps, per `segmentation_token.md`.

This is the fix for a second, independent problem `lean_constrained_beam_search`
inherited from every other decoding mode in this codebase: `<step>` is not a
token the base model has ever been trained to produce -- every mode gets it
out of the model only by *instructing* it ("write ONE step, then the marker
`<step>`, then stop") plus a worked few-shot demonstration
(`DEFAULT_SYSTEM_PROMPT`/`DEFAULT_FEW_SHOT`, `rllf/generation/base_model.py`).
That is an unnatural generation objective layered on top of the distribution
mismatch already documented in `experiments_v1.md` §8 and
`lean-constrained-beam-search.md` §1.

This script instead teaches `<step>`/`<final>` as genuine special tokens via
light supervised fine-tuning (LoRA, via `peft` -- a new dependency for this
codebase, chosen because it is cheap and standard for exactly this "add a
new token + a narrow behavioural SFT" use case; everything else here is
plain PyTorch/`transformers`, matching this project's existing hand-rolled
training style in `rllf/policy/training.py`), on FaithformBench's own
naturally-segmented, unperturbed step sequences -- see
`rllf/data/step_sft.py` for the dataset construction and, importantly, the
leakage boundary: only the 20% "fit" split `stage_fit` uses to train reward
models is ever touched here; the 80% held-out split every AUC/real-Lean
number in this project is measured against is never seen by this script.

    python3 scripts/finetune_step_native.py \\
        --base-model microsoft/Phi-3.5-mini-instruct \\
        --data-root "~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3" \\
        --out /scratch/users/iacopog/rllf-runs/step_native_sft \\
        --epochs 3 --seed 0 --device cuda

Output: a standalone, merged (no LoRA adapter needed at inference time)
Hugging Face checkpoint at `--out` -- a drop-in `model_name_or_path` for
`HFBaseModel`, no other code changes required to use it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rllf.data.faithform import (
    ALL_MODELS,
    LabelPolicy,
    build_labels,
    contaminated_problem_ids,
    load_steps,
    processbench_problem_text,
    to_trajectories,
)
from rllf.data.step_sft import StepSFTExample, build_step_sft_examples
from rllf.protocol import FINAL_TOKEN, SPECIAL_TOKENS, STEP_TOKEN


def verify_single_token(tokenizer, token: str) -> int:
    """segmentation_token.md point 8: the whole point of a dedicated special
    token is that it is *one* tokenizer id, not a multi-subword string --
    verify that before doing anything else, so a boundary detector can later
    be a plain `token_id == step_token_id` check."""
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(
            f"{token!r} tokenizes to {len(ids)} ids ({ids}), not 1 -- "
            f"add_special_tokens should have made this atomic. Aborting: "
            f"a multi-token boundary defeats the point of this fine-tuning."
        )
    return ids[0]


def build_dataset(args: argparse.Namespace) -> List[StepSFTExample]:
    table = load_steps(args.data_root, include_perturbed=False)
    # `to_trajectories` reads a `label` column, but this script only uses the
    # step *text* and *order* from the result (`step_sft.py` never touches
    # `StepRecord.verified`/`.reward`) -- any policy labels the table
    # consistently; `all8_majority` verifiability matches what the rest of
    # this project fits its reward models on.
    table = build_labels(table, LabelPolicy(models=ALL_MODELS, rule="majority", name="all8_majority"))
    text = processbench_problem_text()
    contaminated = contaminated_problem_ids(text)
    trajectories = to_trajectories(table, text, exclude_problems=contaminated)
    print(f"{len(trajectories)} unperturbed FaithformBench trajectories "
          f"({len(contaminated)} MATH-500-overlapping problems already excluded)", flush=True)
    examples = build_step_sft_examples(trajectories, eval_fraction=args.eval_fraction, seed=args.seed)
    print(f"{len(examples)} SFT examples in the fit split (eval_fraction="
          f"{args.eval_fraction}, seed={args.seed} -- must match `stage_fit`'s own split "
          f"for whatever reward-model checkpoint this base model will be decoded with)", flush=True)
    return examples


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default="microsoft/Phi-3.5-mini-instruct")
    p.add_argument("--data-root", default="~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3")
    p.add_argument("--out", required=True)
    p.add_argument("--eval-fraction", type=float, default=0.80,
                   help="must match the reward model's own `stage_fit --seed` split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true", help="build the dataset and print a sample, no training")
    args = p.parse_args(argv)

    examples = build_dataset(args)
    if not examples:
        print("no SFT examples -- nothing to train on", file=sys.stderr)
        return 1
    print("\n--- sample example ---")
    print(examples[0].prompt + examples[0].completion)
    print("---\n", flush=True)
    if args.dry_run:
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.base_model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model.to(args.device)

    added = tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    model.resize_token_embeddings(len(tokenizer))
    print(f"added {added} new special tokens: {SPECIAL_TOKENS}", flush=True)
    step_id = verify_single_token(tokenizer, STEP_TOKEN)
    final_id = verify_single_token(tokenizer, FINAL_TOKEN)
    print(f"  {STEP_TOKEN} -> token id {step_id}, {FINAL_TOKEN} -> token id {final_id} "
          f"(both single tokens, as required)", flush=True)

    # LoRA everywhere, but let the new tokens' rows in the embedding / lm_head
    # matrices train in full -- segmentation_token.md point 7: "make sure the
    # new token's output embedding is trained as well as its input
    # embedding." A frozen (vanilla-LoRA) embedding matrix would leave a
    # freshly-added token at its random initialisation forever.
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize once: prompt + completion + eos, labels masked (-100) over the
    # prompt so the loss only ever scores the model's own generated tokens --
    # standard SFT masking, hand-rolled here (no `trl`) to match this
    # project's existing training code (`rllf/policy/training.py`) rather
    # than pull in a second new training-framework dependency.
    encoded = []
    for ex in examples:
        prompt_ids = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(ex.completion, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = (prompt_ids + completion_ids)[: args.max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: args.max_length]
        if len(input_ids) <= len(prompt_ids):
            continue  # truncated away entirely -- nothing to learn from
        encoded.append((input_ids, labels))
    print(f"{len(encoded)}/{len(examples)} examples kept after tokenization/truncation "
          f"(max_length={args.max_length})", flush=True)

    optimizer = torch.optim.AdamW(
        [p_ for p_ in model.parameters() if p_.requires_grad], lr=args.learning_rate
    )
    model.train()
    started = time.perf_counter()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for input_ids, labels in encoded:
            input_tensor = torch.tensor([input_ids], device=args.device)
            label_tensor = torch.tensor([labels], device=args.device)
            output = model(input_ids=input_tensor, labels=label_tensor)
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            total_loss += float(output.loss)
        print(f"epoch {epoch}: mean loss = {total_loss / len(encoded):.4f} "
              f"({time.perf_counter() - started:.0f}s elapsed)", flush=True)

    # Merge LoRA back in and save a *standalone* checkpoint: a drop-in
    # `model_name_or_path` for `HFBaseModel`, no adapter-loading code needed
    # anywhere else in this codebase.
    print("merging LoRA weights and saving a standalone checkpoint...", flush=True)
    merged = model.merge_and_unload()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    (Path(args.out) / "step_native_sft_meta.json").write_text(json.dumps({
        "base_model": args.base_model, "n_examples": len(encoded), "epochs": args.epochs,
        "eval_fraction": args.eval_fraction, "seed": args.seed,
        "step_token_id": step_id, "final_token_id": final_id,
    }, indent=2))
    print(f"saved to {args.out}", flush=True)

    # Built-in sanity check: does the fine-tuned model actually emit <step>
    # on a fresh prompt, unprompted by any "write ONE step" instruction?
    merged.eval()
    sample_prompt = examples[0].prompt
    inputs = tokenizer(sample_prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        generated = merged.generate(**inputs, max_new_tokens=200, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
    completion = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=False)
    print(f"\n--- sanity check: greedy completion of the first prompt ---\n{completion}\n---", flush=True)
    print(f"emits {STEP_TOKEN!r}: {STEP_TOKEN in completion}   "
          f"emits {FINAL_TOKEN!r}: {FINAL_TOKEN in completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
