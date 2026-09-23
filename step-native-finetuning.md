# Step-native fine-tuning: teaching `<step>`/`<final>` as real tokens

A second, independent fix alongside `lean_constrained_beam_search.md` — that
document's own §1 already names this as a second problem: `<step>` is not a
token any base model has ever been trained to produce. Every decoding mode
in this codebase gets it out of the model only by *instructing* it ("write
ONE step, then the marker `<step>`, then stop", `DEFAULT_SYSTEM_PROMPT` in
`rllf/generation/base_model.py`) plus a worked few-shot demonstration. This
document records the fine-tuning built to fix that directly, following
`segmentation_token.md`.

**Status: implemented, dry-run-verified against the real FaithformBench data,
and now actually trained.** Training completed in under 2 minutes of GPU
time (0.03 GPU-hours: 226 examples, 3 epochs, LoRA on Phi-3.5-mini-instruct):
loss fell 0.496 → 0.290 → 0.150 across the three epochs, and the built-in
sanity check confirms the saved, merged checkpoint emits both `<step>` and
`<final>` naturally, with no instruction telling it to:

```text
There are 9 red sticks. <step>
The number of blue sticks is 5 more than the number of red sticks. <step>
The number of yellow sticks is 3 less than the number of blue sticks. <step>
Number of blue sticks = 9 + 5 = 14. <step>
Number of yellow sticks = 14 - 3 = 11. <step>
Total number of sticks = number of red sticks + number of blue sticks + number of yellow sticks. <step>
Total number of sticks = 9 + 14 + 11 = 34. <step>
<final>  Therefore, the total number of sticks they have is 34.<|endoftext|>
```

**Caveat this result doesn't clear on its own:** the sanity-check prompt was
the *first training example itself*, generated greedily — after 3 epochs on
only 226 examples, reproducing a training prompt this cleanly is expected
and is not yet evidence of generalization to a genuinely unseen problem.

**That generalization check was run** (5 fresh MATH-500 problems, through
`lean_constrained_beam_search`, `beam_width=3`, `expansion_per_beam=2`,
`max_steps=6`) and the result is genuinely informative, not a clean win:

- **Step quality/atomicity is visibly improved.** Compare a hard-problem
  excerpt against the vanilla base model on the *same* two problems
  (`lean-constrained-beam-search.md` §7): vanilla produces one long,
  multi-clause sentence per step; the step-native model produces clean,
  atomic sub-steps —

  ```text
  196 = 14^2.
  A prime factorization of 196 is 2^2 · 7^2.
  For a number N = p1^a1 · p2^a2 · ... · pk^ak, the number of positive
    divisors is (a1+1)(a2+1)...(ai+1).
  For 196 with exponents 2 and 2 for primes 2 and 7 respectively, the
    number of divisors is (2+1)(2+1).
  (2+1)(2+1) = 3·3 = 9.
  Therefore, 196 has 9 positive whole-number divisors.
  ```

  — exactly the FaithformBench-style granularity the fine-tuning targeted,
  on a problem the model was never trained on.

- **But 0/5 reached a final answer within the step budget** (vanilla: 1/5).
  Finer-grained steps is a double-edged outcome for a *fixed* `max_steps`:
  the same reasoning now takes more steps to complete, so a budget sized for
  the old, chunkier granularity binds harder, not less. `max_steps` needs
  re-tuning upward for this checkpoint, not carried over unchanged from
  vanilla-model runs.
- **A second, distinct finding**: the 196-divisors example above reaches a
  complete, correct conclusion ("Therefore, 196 has 9 positive
  whole-number divisors.") on step 6, in substance — but the model emitted
  a plain `<step>` there instead of `<final>`, so `lean_constrained_beam_search`
  never recognised it as done. The model can produce final-answer-quality
  content without reliably *signalling* it via `<final>` at that exact
  juncture — plausibly needs more training signal (227 examples / 3 epochs
  is small) for `<final>`'s trigger condition specifically, since (per §3)
  it only ever appears once per example, at the very end, versus `<step>`
  appearing many times per example.

Full output: `/scratch/users/iacopog/rllf-runs/beam_search_smoke/classifier_mode_smoke_stepnative.json`.

---

## 1. What was built

| Piece | File | What |
| --- | --- | --- |
| The new special token | `rllf/protocol.py` | `FINAL_TOKEN = "<final>"`, added to `SPECIAL_TOKENS` alongside the existing `STEP_TOKEN`; `contains_final_answer`/`extract_final_answer` recognise it |
| The SFT dataset builder | `rllf/data/step_sft.py` (new file) | `render_step_sft_example`, `split_trajectories_by_problem`, `build_step_sft_examples` |
| The fine-tuning script | `scripts/finetune_step_native.py` (new file) | loads a base model, adds the special tokens, resizes embeddings, LoRA-trains on the SFT set, merges, saves a standalone checkpoint |
| Tests | `tests/test_step_sft.py` (new), additions to `tests/test_protocol.py` | 11 new tests, all passing, pure logic (no GPU, no real FaithformBench data needed) |

## 2. The token (request/doc point 1)

`FINAL_TOKEN = "<final>"` joins the existing `STEP_TOKEN = "<step>"` in
`rllf/protocol.py`'s `SPECIAL_TOKENS` tuple — the same tuple
`finetune_step_native.py` passes to `tokenizer.add_special_tokens(...)`, so
both get registered as genuinely new, single-token special tokens (not a
string that happens to tokenize into several ordinary subwords) and get
their own embedding rows via `model.resize_token_embeddings(len(tokenizer))`.
`verify_single_token()` in the fine-tuning script asserts each one encodes
to exactly one token id before training starts (`segmentation_token.md`
point 8) rather than silently proceeding if that assumption is wrong.

`contains_final_answer`/`extract_final_answer` (`rllf/protocol.py`) were
extended to also recognise `<final>`, additively — every existing form
(`Final answer:`, `\boxed{...}`, `####`) still works exactly as before, so
nothing in the existing decoding pipeline needed to change to be compatible
with a model that now sometimes emits `<final>` instead.

## 3. The dataset (request: build it from FaithformBench, fit split only)

`rllf/data/step_sft.py` turns FaithformBench's own naturally-segmented,
**unperturbed** step spine (`rllf.data.faithform.load_steps(...,
include_perturbed=False)`, via the already-existing `to_trajectories`
helper) into `(prompt, completion)` SFT pairs — exactly the "teach the
natural segmentation language by training on complete, correctly-segmented
trajectories" recommendation (`segmentation_token.md` point 4), using real,
already-correct solutions rather than anything synthesised.

**The leakage boundary (the explicit "NOT THE TEST PART" instruction):**
`split_trajectories_by_problem` reproduces `scripts/run_faithform.py`'s own
`split_examples` by-problem split, applied to whole trajectories, with the
**same `(eval_fraction, seed)`** a reward model's own `stage_fit` run used.
`build_step_sft_examples` only ever draws from the *fit* half (20% by
default) — the 80% held-out half, which every AUC and real-Lean number in
this project's reports is measured against, is never touched. This mirrors
exactly how `stage_fit` itself never trains the reward model on that held-out
half — one consistent split boundary for every component trained on
FaithformBench, reward model and base model alike. On the real data (default
settings): **1,133 unperturbed, non-MATH-500-contaminated trajectories →
227 in the fit split** (the rest is the protected held-out portion).

**Prompt** (`SFT_SYSTEM_PROMPT`): `segmentation_token.md`'s own recommended
wording, introducing both tokens and explicitly *not* instructing "one step
and stop" — the doc's central point 4/10, and the actual instruction being
replaced.

**A real finding from actually running this, not an assumption:** the doc's
own worked example puts a short, separate answer line after `<final>`
(`<step>...<step>\n<final> 97`), distinct from the last reasoning step.
FaithformBench's step text has no such separate clean-answer field, so the
first implementation instead used `<final>` only when the trajectory's last
step already looked like an answer (`contains_final_answer`: contains
`\boxed{...}`, `####`, etc.). **Checked against the real 227-example fit
split, that fired on only 2 of them** — FaithformBench's own segmentation
mostly does not preserve a literal answer marker on the final step, even
though it *is* the conclusion. That would have given `<final>` essentially
no training signal. Fixed by marking the trajectory's **last step
unconditionally** as the `<final>` segment, regardless of its wording — still
matches the doc's structural intent ("`<final>` = the whole trajectory is
complete, an answer follows", which is always the *last* segment) even
though the exact worked-example shape differs. A prose-y last step (e.g.
"Therefore, the total number of sticks they have is 34.") still resolves
correctly downstream via `NumericChecker`'s existing last-number fallback,
the same tolerance already relied on elsewhere for prose final answers.

Example (real output, from the actual 227-example fit split):

```text
There are 9 red sticks. <step>
The number of blue sticks is 5 more than the number of red sticks. <step>
The number of yellow sticks is 3 less than the number of blue sticks. <step>
Blue sticks = 9 + 5 = 14. <step>
Yellow sticks = 14 - 3 = 11. <step>
Total sticks = red + blue + yellow. <step>
Total sticks = 9 + 14 + 11 = 34. <step>
<final> Therefore, the total number of sticks they have is 34.
```

## 4. Training (requests: whole-trajectory SFT, `<step>` != EOS)

`scripts/finetune_step_native.py`:

- Trains on **complete solutions** (`segmentation_token.md` point 4), not
  step-by-step conditional pairs — the SFT target is the whole rendered
  trajectory in one completion, so the model learns the natural joint
  distribution over an entire segmented solution rather than an artificial
  "predict one step, then stop" conditional.
- **`<step>` is never EOS** (point 5): the completion is `... <step> ...
  <step> <final> last_step <eos>` — the model only ever sees end-of-sequence
  after the whole trajectory, `<step>` mid-sequence throughout training.
- **LoRA via `peft`** (a new dependency for this codebase — everything else
  here is plain PyTorch/`transformers`, matching this project's existing
  hand-rolled training style in `rllf/policy/training.py`; `peft` was
  chosen specifically because it is cheap, fast, and standard practice for
  exactly this "add a token + narrow behavioural SFT" use case, and 227
  examples over a few epochs does not need a full fine-tune). Deliberately
  targets `modules_to_save=["embed_tokens", "lm_head"]` alongside the LoRA
  adapters on every other linear layer (`target_modules="all-linear"`) —
  point 7's "make sure the new token's output embedding is trained as well
  as its input embedding": a vanilla LoRA setup freezes the embedding
  matrix entirely, which would leave a freshly-added token at its random
  initialisation forever.
- Loss is masked to the completion only (prompt tokens get label `-100`) —
  standard SFT masking, hand-rolled (batch size 1, no padding/attention-mask
  complexity) rather than pulling in `trl` for it, again matching this
  project's existing preference for explicit, dependency-light training code.
- **Output is a standalone, merged checkpoint** (`model.merge_and_unload()`
  then `save_pretrained`), not a LoRA adapter — a drop-in
  `model_name_or_path` for `HFBaseModel`. No other code in this repository
  needs to change to decode with it.
- A built-in sanity check at the end: greedy-generates a completion for one
  training prompt and reports whether `<step>`/`<final>` actually appear —
  a first, cheap correctness signal before any real evaluation.

## 5. Usage

```bash
python3 scripts/finetune_step_native.py \
    --base-model microsoft/Phi-3.5-mini-instruct \
    --data-root "~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3" \
    --out /scratch/users/iacopog/rllf-runs/step_native_sft \
    --epochs 3 --seed 0 --device cuda
```

`--eval-fraction`/`--seed` must match whatever a reward model's own
`stage_fit --seed` used, so the base model and reward model were trained on
the identical fit/held-out boundary. `--dry-run` builds and prints the
dataset without loading any model or touching the GPU — use it to sanity
check the data before spending compute (this is how the 227-example count
and the `<final>` fix above were actually found, no GPU involved).

Once trained, the checkpoint at `--out` is used exactly like any other base
model:

```python
base_model = HFBaseModel("/scratch/users/iacopog/rllf-runs/step_native_sft", ...)
```

— including as the `base_model` argument to
`rllf.generation.beam_search.lean_constrained_beam_search`, which is the
actual motivation for building this: `<step>` becomes a boundary the model
produces because it has learned to, not because a stop-string cut it off
mid-thought.

## 6. What's not yet done

- **`max_steps` needs re-tuning for this checkpoint.** §5's 5-problem
  generalization check found the step-native model's finer step granularity
  means the same `max_steps=6` that vanilla Phi-3.5-mini-instruct used now
  binds *harder* (0/5 reached an answer, vs. 1/5 for vanilla) — not a defect
  in the fine-tuning, a budget sized for the old granularity. A rerun with a
  larger `max_steps` (and/or `max_tokens_per_step`) is the natural next
  check, not yet done.
- **`<final>`'s trigger reliability.** §5 found at least one case (the
  196-divisors example) where the model reached a complete, correct
  conclusion in substance but emitted `<step>` rather than `<final>` at that
  point. `<final>` appears once per training example vs. `<step>`'s many —
  plausibly under-trained relative to `<step>` at only 227 examples / 3
  epochs. Worth checking whether more epochs, more data, or a
  higher-weighted loss on the `<final>` token position fixes this before
  concluding it's a fundamental limit.
- **No accuracy measurement yet.** §5's 5-problem sample (0/5 correct either
  way) is far too small to say anything about whether this fine-tuning
  helps or hurts actual MATH-500 accuracy at scale — it was sized to check
  mechanism and step quality, not to measure accuracy.
- **A single base model.** Only wired up for `microsoft/Phi-3.5-mini-instruct`
  so far (`--base-model` is a flag, so any causal LM `transformers` and
  `peft`'s `"all-linear"` target-module resolution support should work, but
  only Phi-3.5-mini has actually been run).
- **Held-out fit-split size.** 227 trajectories is a small SFT set. The doc
  argues this should be fine ("the token appears frequently and in a highly
  predictable structural context" — point 7), but that is an expectation,
  not yet a measured outcome.
