# The ProcessBench experiment

Real data, a real frozen base model, real Lean verification. This document
records the setup, the measurements, and the limits that the measurements
expose.

    python scripts/run_processbench.py --stage all --out runs/pb

## 1. Dataset

[`Qwen/ProcessBench`](https://huggingface.co/datasets/Qwen/ProcessBench) is a
step-level benchmark: model-written solutions to maths problems, each annotated
with `label` (index of the first erroneous step, `-1` if clean) and
`final_answer_correct`.

We keep only the rows with **`final_answer_correct=True`**. Those are the rows
whose last step carries a *trusted* answer, which is what the outcome reward
needs — ProcessBench ships no separate ground-truth answer field.

| subset | rows | `final_answer_correct` | answer recoverable |
| --- | --- | --- | --- |
| gsm8k | 400 | 200 | 188 boxed + 12 last-number |
| math | 1000 | 500 | 446 boxed + 54 last-number |
| olympiadbench | 1000 | 500 | 453 boxed + 46 last-number (1 lost) |
| omnimath | 1000 | 500 | 445 boxed + 44 last-number (11 lost) |

Answer recovery (`rllf/data/processbench.py`) takes the last `\boxed{...}` from
the solution, matched with a **brace counter** rather than a regex — MATH
answers nest braces (`\boxed{\frac{1}{2}}`) and a non-greedy match truncates at
the first inner `}`. Failing that it falls back to the last number in the final
step.

**Splitting is by problem text, not by row.** ProcessBench contains several
solutions to the same problem (375 distinct problems across 400 gsm8k rows), so
a row-level split would put the same question on both sides of the experiment.

The headline experiment uses **gsm8k**: 198 unique problems → **40 validation /
158 test**, disjoint. Section 5 explains why not the harder subsets.

## 2. What each split is for

* **Validation (20%)** — the frozen base model proposes its *own* solutions.
  Every step is auto-formalized and checked in Lean; every final answer is
  checked against the recovered reference. The result is the dataset
  `D = {(q, τ, r_step, r_outcome)}` that trains the process reward model.
  ProcessBench's own annotated steps are used **only** to recover the reference
  answer — never as training targets.
* **Test (80%)** — held out. The base model and the policy-guided decoder are
  compared on it, and the policy has never seen these questions.

## 3. Components

| role | choice | note |
| --- | --- | --- |
| base model `M` | `microsoft/Phi-3.5-mini-instruct` (3.8B, bf16, MPS) | **frozen**; never fine-tuned |
| policy `π_φ` encoder | `bert-base-uncased` + step/outcome heads | the only trained component |
| step reward | rule-based formalizer → Lean 4.33.1 | binary, no mathlib |
| outcome reward | exact / numeric / symbolic ladder | binary |

An un-finetuned model does not emit `<step>` from an instruction alone. It takes
a worked few-shot demonstration, plus adapter-level normalisation of the
boundaries the model actually produces (a blank line, a `Step 2:` heading) onto
the one protocol boundary. That translation lives in the base-model adapter, so
the dataset, verifier and policy all still see a single convention.

## 4. The validation dataset

40 problems × 3 samples × 2 temperatures (0.7, 1.0) = **240 trajectories, 859
steps**, ~57 s/problem including Lean.

| | |
| --- | --- |
| step reward = 1 | 652 (75.9%) |
| step reward = 0, formalization failed | 197 |
| step reward = 0, Lean rejected | 10 (audited: **0 genuine arithmetic errors**, see 6.5) |
| outcome reward = 1 | 79.2% |

The label mix is workable but worth reading carefully: **most negatives are
"this step is not an extractable arithmetic claim", not "this step is wrong".**
So the step reward is largely a *formalizability* signal on this subset. That is
a property of the data, not a defect of the pipeline — and it predicts which
metric guidance should move (section 6).

## 5. How far auto-formalization actually reaches

Before any policy is trained, it is worth measuring the reward signal itself.
`scripts/formalization_reach.py` runs the verifier over ProcessBench's *own*
annotated-correct solutions — no language model involved. A step of a solution
ProcessBench marks correct *should* verify.

| subset | steps | formalization | verification | verified given formalized |
| --- | --- | --- | --- | --- |
| gsm8k | 219 | 0.379 | 0.329 | **0.867** |
| math | 277 | 0.440 | 0.242 | 0.549 |
| olympiadbench | 359 | 0.490 | 0.242 | 0.494 |
| omnimath | 315 | 0.413 | 0.222 | 0.538 |

Residual failure categories:

| subset | no claim | verified | parse error | unsupported | unsolved | elaboration | refuted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gsm8k | 129 | 72 | 0 | 7 | 3 | 8 | 0 |
| math | 133 | 67 | 11 | 22 | 19 | 19 | 6 |
| olympiadbench | 149 | 87 | 53 | 34 | 28 | 5 | 3 |
| omnimath | 164 | 70 | 29 | 21 | 19 | 8 | 4 |

Three things this says:

1. **Auto-formalization reaches 38-49% of real reasoning steps.** The dominant
   failure everywhere is `no_claim_found`: prose like *"Let's break this down"*
   or *"Therefore the red rope is 20 centimeters"* carries no relational claim.
   A binary reward scores these 0, conflating "unformalizable" with "wrong" --
   which is why `formalization_success_rate` is reported separately, and why
   masking these out of the step loss (rather than labelling them negative) is
   the first improvement to make.
2. **Step history matters more than expected.** Verifying each step against the
   steps before it -- so the formalizer can bind a variable introduced earlier
   -- raised coverage on olympiadbench from 0.337 to 0.490 and cut
   `unsupported_claim` from 89 to 34. An earlier version of this script passed
   an empty history and badly understated the method's reach.
3. **Faithfulness, not coverage, is what rules out the harder subsets.** On
   gsm8k 87% of formalized steps verify. On competition maths it is 49-55%:
   those solutions are correct, so roughly half the formalizations are saying
   something the prose did not. Training on that is training on ~50% label noise
   in the positive class.

### A cautionary measurement

An earlier revision of the formalizer treated *any* juxtaposition as
multiplication, so `f(2x+3) = 2f(x) + 3` became `f * (2*x+3) = 2*f*(x) + 3`.
That is not a step that fails loudly -- it elaborates cleanly and sometimes
*proves*, while meaning something entirely different.

It made the headline numbers look better. Restricting the rule to numeric left
operands (`4B`, `4(3)`, `(a)(b)`, never `f(x)`) **lowered** apparent faithfulness
on olympiadbench from 0.574 to 0.494 and on omnimath from 0.662 to 0.538, moving
those steps into `parse_error` where they belong. gsm8k and math were unaffected
-- they contain no function notation -- so the headline experiment never
depended on the bug.

The lesson generalises past this one regex: **Lean cannot tell you when a
formalization is faithful, only when it is provable.** Any scaling step that
raises formalization volume -- a neural autoformalizer above all -- needs an
independent faithfulness check (back-translation, or held-out human-checked
pairs) tracked as a regression metric, or coverage gains will silently buy label
corruption.

This is why gsm8k carries the headline experiment. Running it on olympiadbench
would mostly measure the formalizer's coverage gap, not the method.

## 6. Results

### 6.1 The policy

Trained on the 240 validation trajectories (785 train / 295 dev states, split by
question), `bert-base-uncased` + step and outcome heads, early-stopped on
held-out step AUC, then refined with the listwise objective and calibrated on
the dev split:

| head | dev AUC | dev accuracy |
| --- | --- | --- |
| step (will it formalize and verify in Lean?) | **0.956** | 0.850 |
| outcome (will the answer be right?) | 0.687 | 0.810 |

The step head learns its target well. The outcome head does not: its 0.810
accuracy merely matches the majority class, and predicting final-answer
correctness from a *prefix* is both intrinsically harder and much thinner on
data (295 dev states from 40 problems).

### 6.2 Held-out decoding, all 158 test problems

`K = 4` candidate steps, `alpha = 1.0`, deferred Lean verification for every
method so process metrics are comparable.

| method | answer accuracy | step verification | fully verified | formalization | steps | s/problem |
| --- | --- | --- | --- | --- | --- | --- |
| `base_greedy` | 0.709 | 0.774 | 0.456 | 0.795 | 3.83 | 10.7 |
| `random_of_k` | 0.696 | 0.795 | 0.538 | 0.815 | 3.76 | 35.6 |
| **`policy_guided`** | **0.753** | **0.877** | **0.696** | **0.904** | 3.55 | 33.5 |

`random_of_k` is the compute-matched control: the same four candidates per step,
chosen at random. It separates what the *policy* contributes from what the extra
sampling contributes.

**Process quality improves decisively.** Step verification rises 0.774 → 0.877
(+0.103) and fully-verified trajectories 0.456 → **0.696**, a 53% relative
increase. The policy was trained to predict formalize-and-verify, and that is
precisely the axis that moves. Guided decoding also reaches its answer in fewer
steps (3.55 vs 3.83).

**Answer accuracy improves, but not significantly.** +0.044 over greedy, from 17
wins against 10 losses — two-sided exact McNemar **p = 0.248**. Against the
compute-matched control it is 16-7, **p = 0.093**. Both trend the right way,
neither clears conventional significance. At n = 158 this experiment simply
lacks the power to resolve a four-point difference; the honest reading is
*process quality improves clearly, answer accuracy is inconclusive*.

**The extra sampling alone buys nothing.** `random_of_k` costs 3.3x the latency
and lands *below* greedy on answers (0.696 vs 0.709). The gains above are
attributable to the reward model's ranking, not to the larger candidate pool.

### 6.3 Does the reward model generalise?

Re-scoring the steps of the test trajectories against their recorded Lean
labels — questions the policy never saw:

| trajectories scored | step AUC | step ECE | outcome AUC | outcome ECE |
| --- | --- | --- | --- | --- |
| `base_greedy` | 0.924 | 0.018 | 0.594 | 0.291 |
| `random_of_k` | 0.944 | 0.043 | 0.656 | 0.312 |
| `policy_guided` | 0.892 | 0.024 | 0.624 | 0.245 |

The step head holds up on held-out problems (0.89-0.94 against 0.956 on the dev
split) and is **well calibrated** (ECE 0.018-0.043), which is what makes it safe
to feed to the decoder as a log-probability. The outcome head does not transfer
(AUC 0.59-0.66, ECE 0.25-0.31) — it should carry little weight until it is
trained on far more problems, and `--outcome-weight 0` is the conservative
setting.

### 6.5 What the step reward actually measured — audit

Nothing is excluded anywhere in this pipeline. Every step of every trajectory
becomes one training example; a step the extractor cannot express is labelled
**0**, not dropped. 47.5% of training trajectories and 53.8% of `base_greedy`
test trajectories contain at least one such step, and all are kept. The only
masked label is the final-answer state, which is not a Lean step.

That design decision is what the following three measurements are about.

**(a) The gain is entirely in formalization, not correctness.** Decomposing the
step-verification rate into coverage and conditional pass-rate:

| method | verified | formalized | verified given formalized |
| --- | --- | --- | --- |
| `base_greedy` | 0.774 | 0.795 | 0.973 |
| `random_of_k` | 0.795 | 0.815 | 0.975 |
| `policy_guided` | 0.877 | 0.904 | **0.970** |

The conditional rate is flat -- slightly *down* under guidance. The whole
+0.103 comes from the extractor firing more often (`no_claim_found` steps drop
from 119 to 47). Guidance made the reasoning more explicitly arithmetic; it did
not make the arithmetic more correct.

**(b) A regex beats the learned model at its own task.** Scoring the 1760
held-out test states:

| predictor | AUC | accuracy |
| --- | --- | --- |
| "contains a digit and a relation symbol" | 0.761 | 0.911 |
| **the formalizer's own extractor, as a predictor** | **0.961** | **0.977** |
| the learned 110M-parameter process reward model | 0.892-0.944 | 0.932-0.947 |

The extractor is very nearly the label-generating function, so this is expected
-- but it is the correct control, and it says the 0.956 AUC is evidence that
BERT can imitate a regex, not evidence of learned mathematical judgement.

**(c) Every Lean rejection was a false negative.** Auditing all 46 distinct
`lean_check_failed` steps across every run:

| cause | count |
| --- | --- |
| a prose word split the expression (`3 books * 2` → `(* 2) = 6`) | 9 |
| model rounded (`8/1.5 = 5.33`); Lean uses exact rationals | 9 |
| leading decimal lost (`$.05` → `05`) | 4 |
| prose fragment read as a variable (`that's` → `s`, `km`) | 3 |
| hypothesis selection or other formalizer artefact | 21 |
| **genuine arithmetic error the model made** | **0** |

Not one. On gsm8k, Phi-3.5-mini rarely makes *local* arithmetic slips; its
failures are wrong set-ups, which are not refutable at the level of a single
step. So the `verify` half of "formalize and verify" contributed no usable
signal here, and contributed some label noise.

### 6.6 What this does and does not show

**Established.** The end-to-end machinery works: ProcessBench → sampled
trajectories → Lean labels → a trained reward model → guided decoding → a
measurable, reproducible shift in the generated reasoning, with no Lean calls at
decode time. Guided decoding raises the share of steps that state explicit,
self-contained arithmetic from 0.795 to 0.904, and fully-verified trajectories
from 0.456 to 0.696.

**Not established.**

* That the reward model learned anything about *mathematics*. Its own extractor
  outscores it (6.5b), so on this data it is a formalizability classifier.
* That Lean contributed to the reward. Every rejection audited was a false
  negative (6.5c); removing Lean entirely would have changed almost no label.
* That answer accuracy improves. +0.044, exact McNemar p = 0.248.
* Anything about harder mathematics: one subset, one base model, 158 test
  problems, one seed.

**The honest summary** is that this validates the *pipeline* and refutes the
*reward* on this subset. The step reward needs a regime where Lean can disagree
with the extractor -- richer formalization (so coverage is not the binding
constraint), tolerance for rounding, and problems whose steps contain refutable
local claims. Sections 5 and 7 of the README's scaling notes address exactly
that.


## 7. Bugs this experiment surfaced

Running on real data found defects that the synthetic pipeline never exercised:

| bug | consequence had it shipped |
| --- | --- |
| `HFBaseModel` never checked for MPS | every generation ran on CPU |
| No `<step>` from an instruction alone | trajectories would not segment |
| Outcome checker scored *"Anne is 8 years old."* vs `8` as **wrong** | false-negative outcome labels, corrupting both training targets and the headline metric |
| Implicit multiplication (`4B`, `4(3)`) rendered invalid Lean | dominant `parse_error` on competition maths |
| Lean cache salt keyed on formalizer *class name* | stale verdicts silently reused across a formalizer fix |
| HF tokenizer truncated from the right | the current step — the thing being scored — cut off, leaving the policy to score context |
| Best-checkpoint snapshot kept on the accelerator | a full model copy pinned per improving epoch |

## 8. Reproducing

```bash
python scripts/run_processbench.py --stage data  --out runs/pb    # ~40 min
python scripts/run_processbench.py --stage train --out runs/pb --device cpu
python scripts/run_processbench.py --stage eval  --out runs/pb    # resumable
python scripts/run_processbench.py --stage report   --out runs/pb
python scripts/run_processbench.py --stage diagnose --out runs/pb
python scripts/formalization_reach.py --per-subset 40
```

`--stage eval` checkpoints every 10 problems and skips methods already
complete, so it can be interrupted and resumed. Lean verdicts are cached across
every stage.
