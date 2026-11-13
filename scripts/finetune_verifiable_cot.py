#!/usr/bin/env python3
"""Verifiable chain-of-thought SFT (forked-tree-classifier-retraining.md sec. 13.4).

Training data: trajectories of the MATH-train forked trees (MATH-500 never touched) that are
correct and contain no Python-refuted step; each step carries a verifier tag by the heuristic
    python  if the Python weak verifier verified the step,
    lean    if it did not but the real-Lean pipeline proved it,
    none    otherwise.
The assistant turn is a JSON list of {"step": ..., "verifier": ...} objects followed by one
{"final_answer": ...} object, under a system prompt describing the format (chat template of the
base model; no new tokens). `--tags none` trains the same trajectories with every tag set to
"none" (format control). LoRA (all-linear), merged standalone checkpoint, as finetune_step_native.py.
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rllf.data.forked_tree import load_trees
from rllf.protocol import extract_final_answer

SYSTEM = ("You are solving a mathematics problem. Write the solution as a JSON list. Each element except the last is an "
          "object {\"step\": <one concise reasoning step>, \"verifier\": <\"python\" | \"lean\" | \"none\">}, where the verifier field "
          "says how that step can be checked: \"python\" if a short Python/sympy program can recompute and confirm it, \"lean\" if it is "
          "a mathematical fact that can be formalized and proved in Lean but not computed directly, \"none\" if it is a plan, a "
          "definition, or otherwise not mechanically checkable. The last element is {\"final_answer\": <the final answer>}. "
          "Output only the JSON list.")


def build_from_pool(args):
    """Round 2: trajectories sampled from the round-1 model on MATH-train, kept if correct and no step is
    Python-refuted; tags python (verified) | none; final answer = the short reference answer; at most
    `--per-problem` per problem, most-verified first."""
    verd = json.load(open(args.python_verdicts)); by_q = {}
    from rllf.data.forked_tree import step_key
    for line in open(args.pool):
        r = json.loads(line)
        if not r["correct"]: continue
        ks = [step_key(r["question"], r["steps"][:i], s) for i, s in enumerate(r["steps"])]
        if any(k not in verd for k in ks): continue
        v = [verd[k]["verdict"] for k in ks]
        if any(x == "refuted" for x in v): continue
        tags = ["python" if x == "verified" else "none" for x in v]
        share = tags.count("python") / len(tags)
        if args.tags == "none": tags = ["none"] * len(tags)
        body = [{"step": s, "verifier": g} for s, g in zip(r["steps"], tags)] + [{"final_answer": r["reference"]}]
        by_q.setdefault(r["question"], []).append((share, {"question": r["question"], "completion": json.dumps(body, ensure_ascii=False), "level": r["level"], "n": len(ks), "tags": tags}))
    out = []
    for q, c in by_q.items():
        seen = set()
        for _, e in sorted(c, key=lambda x: -x[0]):
            if e["completion"] in seen: continue
            seen.add(e["completion"]); out.append(e)
            if len(seen) >= args.per_problem: break
    return out


def build(args):
    R = Path("/scratch/users/iacopog/rllf-runs")
    verd = json.load(open(args.python_verdicts))["detail"] if "detail" in json.load(open(args.python_verdicts)) else json.load(open(args.python_verdicts))
    lean = json.load(open(R / "forked_tree" / "real_lean.json"))["detail"]
    trees = load_trees(str(R / "forked_tree" / "samples.jsonl"))
    examples = []
    for t in trees:
        for tr in t.trajectories:
            ks = t.step_keys(tr)
            if not ks or not tr.correct or any(k not in verd or k not in lean for k in ks):
                continue
            v = [verd[k]["verdict"] if "verdict" in verd[k] else ("verified" if verd[k]["proved"] else "unknown") for k in ks]
            if any(x == "refuted" for x in v):
                continue
            tags = ["python" if x == "verified" else ("lean" if lean[k]["proved"] else "none") for x, k in zip(v, ks)]
            if args.tags == "none":
                tags = ["none"] * len(tags)
            final = extract_final_answer(tr.final_answer) or tr.final_answer.replace("<final>", "").strip()
            steps = [s.replace("<final>", "").strip() for s in tr.steps]
            body = [{"step": s, "verifier": g} for s, g in zip(steps, tags)] + [{"final_answer": final}]
            examples.append({"question": t.question, "completion": json.dumps(body, ensure_ascii=False), "level": t.metadata.get("level"), "n": len(ks), "tags": tags})
    return examples


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--python-verdicts", default="/scratch/users/iacopog/rllf-runs/pyverify_30b/verdicts.json")
    p.add_argument("--pool", default=None); p.add_argument("--per-problem", type=int, default=2)
    p.add_argument("--out", required=True); p.add_argument("--tags", default="heuristic", choices=["heuristic", "none"])
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-r", type=int, default=16); p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=1536); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    examples = build_from_pool(args) if args.pool else build(args); random.Random(args.seed).shuffle(examples)
    from collections import Counter
    print(f"{len(examples)} training trajectories over {len({e['question'] for e in examples})} problems; tags: {Counter(g for e in examples for g in e['tags'])}; levels: {Counter(e['level'] for e in examples)}", flush=True)
    print("--- sample ---\n", examples[0]["question"][:200], "\n", examples[0]["completion"][:600], "\n---", flush=True)
    if args.dry_run: return 0
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(args.device)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    encoded = []
    for ex in examples:
        prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["question"]}], tokenize=False, add_generation_prompt=True)
        pi = tok(prompt, add_special_tokens=False)["input_ids"]; ci = tok(ex["completion"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (pi + ci)[: args.max_length]; labels = ([-100] * len(pi) + ci)[: args.max_length]
        if len(ids) > len(pi): encoded.append((ids, labels))
    print(f"{len(encoded)} examples after truncation (max_length {args.max_length})", flush=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.learning_rate); model.train(); t0 = time.perf_counter()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(encoded); tot = 0.0
        for ids, labels in encoded:
            out = model(input_ids=torch.tensor([ids], device=args.device), labels=torch.tensor([labels], device=args.device))
            opt.zero_grad(); out.loss.backward(); opt.step(); tot += float(out.loss)
        print(f"epoch {epoch}: mean loss {tot / len(encoded):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    merged = model.merge_and_unload(); Path(args.out).mkdir(parents=True, exist_ok=True); merged.save_pretrained(args.out); tok.save_pretrained(args.out)
    (Path(args.out) / "verifiable_cot_meta.json").write_text(json.dumps({"base_model": args.base_model, "n_examples": len(encoded), "epochs": args.epochs, "tags": args.tags, "system_prompt": SYSTEM}, indent=2))
    merged.eval(); prompt = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": examples[0]["question"]}], tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad(): g = merged.generate(**inp, max_new_tokens=400, do_sample=False, pad_token_id=tok.pad_token_id)
    print("--- sanity greedy completion ---\n", tok.decode(g[0, inp["input_ids"].shape[-1]:], skip_special_tokens=True)[:800], "\n---", flush=True)
    print(f"saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
