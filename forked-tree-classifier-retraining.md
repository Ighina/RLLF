# Retraining the step classifier on forked trajectory trees

Status: experiment 1 fits done (§7.4); **experiment 2 done — null on formalization and
accuracy** (§5.1–5.3); **`full_trajectory_rerank` plug-in done — null on the 15-problem cache** (§4.1); **Qwen3-1.7B
encoder refit done (§8) — a much better step judge (AUC 0.72 OOD), still no accuracy or
significant formalization gain on easy problems.** α sweep (§8.6), refutation probe (§9),
**Lean-in-the-loop (§10.1: +6 pts verified content, accuracy unchanged)** and **hard-problem
rerank (§11.1: first accuracy gain, +0.075 [+0.013, +0.138])** done; hard beam arms (§11.2) done;
**internal-state probes as judges (§12: residual and attention-head) done — usable, not better.**

## 1. Why

`experiments_v1.md` §10.1–§10.2 diagnosed two independent failures of every
classifier fitted on FaithformBench:

1. **Training problem.** FaithformBench has one trajectory per problem, so the
   supervised BCE fit can only separate *problems* (easy vs. hard) and step
   *surface* (equation density), never *competing steps to the same problem*.
   Measured: score/true-rate correlation is +0.21/+0.25 pooled across problems
   and −0.15/−0.18 *within* a problem, where selection happens. Simpson's paradox.
2. **Distribution problem.** The classifiers' step-level AUC against real Lean
   on steps written by the model we decode (Qwen2.5-7B-Instruct) is 0.50–0.61 for
   every labeler and architecture, vs. 0.79–0.83 on FaithformBench's own held-out
   split. Not the autoformalizer (the Goedel-matched checkpoints do *worse*):
   FaithformBench steps are not our generator's steps.

Both are fixed by the same data: **several solutions to the same problem,
written by the decoded model, each step labelled by the real Lean pipeline**,
with a loss that holds the problem fixed. This document specifies that data and
the two experiments that use it.

## 2. The generator: step-native Qwen2.5-7B-Instruct

`scripts/finetune_step_native.py --base-model Qwen/Qwen2.5-7B-Instruct` — the
recipe of `step-native-finetuning.md` (LoRA `all-linear` + full training of the
`embed_tokens`/`lm_head` rows so the new `<step>`/`<final>` tokens learn; 226
FaithformBench **fit-split** solutions, never the held-out split; 3 epochs).
Checkpoint: `/scratch/users/iacopog/rllf-runs/step_native_sft_qwen7b`.

Decoding it needs `rllf.generation.base_model.StepNativeHFBaseModel`, new here.
The Phi-3.5 smoke test in `lean-constrained-beam-search.md` used the generic
`HFBaseModel` and had two silent mismatches, both fixed by the subclass:

- the prompt was wrapped in the chat template with a `Question:` user turn; the
  SFT trained on the raw `SFT_SYSTEM_PROMPT + "Problem:\n…\n\nSolution:\n"` string
  (`render_step_sft_prompt`, now shared by trainer and decoder);
- `<step>`/`<final>` were added as *special* tokens and the base class decodes
  with `skip_special_tokens=True`, deleting them — `<final> 42` came back as the
  bare step `42`, `contains_final_answer` never fired, and every trajectory ran to
  `max_steps`. That is why the smoke test "reached an answer" 0/5 times. The
  subclass strips only pad/EOS and keeps `<final>` in the returned text.

## 3. The data: forked trajectory trees (`rllf/data/forked_tree.py`)

Per problem, `n_trajectories = 4`:

```
T0:  s1   s2   s3   s4   <final>              root, sampled from scratch
T1:  s1   s2   s3'  s4'  <final>              fork from T0 at k=2
T2:  s1'  s2'' s3''      <final>              fork from T0 at k=0
T3:  s1   s2   s3'  s4'' <final>              fork from T1 at k=3
```

`rllf.generation.forked_sampling.sample_forked_tree`: trajectory 0 is sampled
from scratch; each later one picks a fork point uniformly over every
`(trajectory, step index)` so far, reuses that prefix, samples a **fresh step** at
that index and continues to completion. One sampled step per call
(`generate_step_candidates(num_candidates=1)`, T=0.8, top-p 0.95, ≤256 tokens per
step, no `max_steps` cap — only the 200-step safety ceiling, per
`experiments_v1.md` §9). Every trajectory records its parent, fork index, stop
reason (`final` / `empty` / `max_steps`), correctness (`default_outcome_checker`
against the MATH reference) and whether the re-drawn sibling happened to be
textually identical.

What one tree yields:

| unit | count / problem | used by |
|---|---|---|
| complete trajectories | 4 | experiment 1 (ORM over whole solutions; PRM over prefixes) |
| sibling pairs behind an identical prefix | 3 (one per fork) | experiment 2 (boundary decision), also experiment 1's PRM |
| unique steps to Lean-verify | ~28 instead of ~42 | shared prefixes verified once (`unique_steps`, content-addressed by `step_key(question, history, step)`) |

The saving is what makes one Lean budget serve both experiments: a fork at a
uniform position shares on average half a trajectory, so 3 forks cost ~65% of
3 independent trajectories in Lean calls while adding the sibling pairs that
independent sampling never produces.

**Problems.** 150 from `HuggingFaceH4/MATH` **train** (30 per level, seed 0), so:
disjoint from MATH-500 (which is MATH *test*; checked by text anyway), and
disjoint from every FaithformBench problem text (2,842 excluded) — neither the
SFT's training solutions nor the classifier eval's held-out problems are sampled
on. The reference answer is the solution's last `\boxed{}`; that surfaced a
pre-existing bug — `_BOXED_RE` matched one brace level, so
`\boxed{-\frac{\sqrt{3}}{3}}` fell back to "last line" — now fixed with a
balanced-brace `extract_boxed` in `rllf/protocol.py` (also used by the outcome
checker, so nested boxed *model* answers are now graded correctly too).

**Correctness.** The first trees exposed a second grading gap: the composite
checker's rungs (exact / numeric / sympy-on-Lean-normalised text) agreed only on
numbers or byte-identical strings, so the model's `-√3/3` vs. the reference
`-\frac{\sqrt{3}}{3}` — and `2π/5` vs. `\frac{2\pi}{5}`, `(B)` vs. `\text{(B)}` —
were all graded wrong (4/64 "correct" on the first 16 trees, all four
trajectories of the `\cot(-60°)` problem right). New last rung
`MathEquivalenceChecker` (`rllf/verification/math_equivalence.py`: balanced
`\frac`/`\sqrt` expansion, unicode, value extraction after the last `=` / ` is ` /
`:`, sympy equivalence, element-wise tuples) takes that to 46/64; the rest are
genuine errors (`|(12−9i)(8+15i)| = 249` vs 255) or values buried in prose. The
sampling job predates the rung, so `forked_tree_regrade.py` rewrites `correct`
on the finished file (old value kept in metadata). Note the same rung now applies
to every MATH-500 evaluation: stored trajectories from earlier arms can be
re-graded offline, and nested/LaTeX-answer accuracy will move up slightly.

**Labels.** `forked_tree_real_lean.py`: the same Goedel-Formalizer-V2-8B →
DeepSeek-Prover-V2-7B (n=16) → `lake exe repl` pipeline as every real-Lean number
in this project, over the unique steps only; unlike §10, the Lean statement and
proof are stored per step so failures can later be split into
formalize / prove / genuinely-false. Labels file:
`/scratch/users/iacopog/rllf-runs/forked_tree/real_lean.json`,
keyed by `step_key`.

Both scripts checkpoint (per problem / every 3 steps) and resume unchanged.

## 4. Experiment 1 — full-trajectory rerank with a within-problem classifier

`scripts/train_forked_classifier.py`, four objectives on the identical
by-problem 80/20 tree split (`split_trees`), all evaluated on the held-out
trees by `rllf.evaluation.forked_tree_eval`:

| objective | loss | what it tests |
|---|---|---|
| `supervised` | masked BCE, step head = proved, outcome head = correct | the old target on the new data (control: is it the *data* or the *loss*?) |
| `pairwise` | −log σ(s⁺ − s⁻) over within-problem pairs only: fork siblings (same prefix) + cross-trajectory pairs of the same problem (`build_problem_pairs`) | problem held fixed inside every loss term; **step target unchanged** (proved) |
| `listwise` | softmax CE over each problem's proved steps | same contrast, list-shaped |
| `orm` | outcome head, softmax CE over a problem's 4 whole trajectories, targets ∝ true formalization rate (or `--orm-target correct`) | the outcome reward model for `full_trajectory_rerank` |

Held-out metrics (the ones §10.2 showed the offline AUC does not predict):
step AUC vs. real Lean; sibling-pair accuracy at the forks; and per problem the
trajectory the policy would pick (PRM-style mean step probability and ORM-style
outcome head) — its true formalization rate vs. random (population mean) and
oracle, its correctness, and the within-problem Spearman ρ. `--compare` scores
the existing FaithformBench-trained policies on the same held-out trees, so the
first table answers "did the new data / loss beat 0.535-vs-random, and did the
within-problem ρ turn positive?".

Then the plug-in: the best checkpoint goes into `full_trajectory_rerank` on the
15-problem §10 cache (free, already Lean-labelled) and the 110-problem set.

Budget: sampling ~1 GPU-h; Lean ≈ 150 × 28 unique steps ≈ 4,200 calls ≈ **65 GPU-h**
at the observed 65 steps/GPU-h (n=16 prover samples kept for label
comparability with every earlier number); classifier fits are minutes on MiniLM.

### 4.1 Plug-in on the 15-problem cache (Oct 14) — null, and the classifier does not transfer

`rerank15_real_lean.py` re-labelled the §10 cache (15 MATH-500 problems × 10 natural
Qwen2.5-7B trajectories, retro-segmented; 1,556 unique steps) under the step-level
`facts` rendering with the §7.2 settings (8 prover samples, 60 s, maxHeartbeats 400000):
8.1 + 4.9 GPU-h over two gpuq jobs (the first died with the server at 588 labels; resumed
from checkpoint), 119 steps/GPU-h, **57.6% proved**. Labels
`full_trajectory_rerank/real_lean_facts.json` (statements and proofs stored); the Sep 24
legacy-rendering labels stay in `real_lean.json`. Analysis `rerank15_analyse.py`
(`logs/rerank15_analyse.log`). Each classifier picks, per problem, the trajectory with the
highest mean step probability; random = the problem's mean over its 10 trajectories.

| policy | step AUC | pick | random | oracle | pick − random [95% CI] | pick acc | rand acc | acc − random [95% CI] | within-problem ρ | ρ(score, n steps) |
|---|---|---|---|---|---|---|---|---|---|---|
| `forked_supervised` (§7.4) | 0.572 | 0.622 | 0.606 | 0.806 | +0.015 [−0.079, +0.106] | **0.867** | 0.753 | +0.113 [−0.027, +0.267] | −0.020 | +0.009 |
| old `prm_verifiability_all8_majority` | **0.599** | 0.637 | 0.606 | 0.806 | +0.031 [−0.069, +0.125] | 0.733 | 0.753 | −0.020 [−0.160, +0.133] | +0.226 | −0.199 |
| old `prm_agreement_all8_majority` | 0.546 | 0.609 | 0.606 | 0.806 | +0.003 [−0.080, +0.084] | 0.733 | 0.753 | −0.020 [−0.173, +0.140] | +0.048 | −0.110 |
| §10's cached Qwen3-1.7B scores | — | 0.651 | 0.606 | 0.806 | +0.044 [−0.039, +0.125] | 0.733 | 0.753 | −0.020 [−0.160, +0.120] | +0.112 | −0.295 |

(CIs: 5,000-sample bootstrap over problems; one problem = 0.067. The cached MiniLM scores
of §10 reproduce the `old_verifiability` row exactly, so the scoring path is unchanged.)

Reading:

1. **The forked classifier does not transfer to this generator's natural steps.** Its step
   AUC against real Lean is 0.572 here, below the old verifiability classifier's 0.599, and
   far from the 0.692 it reaches on held-out forked trees (§7.4) or the 0.630 on exp-2's
   step-native MATH-500 steps (§5.3). The steps it was trained on are `<step>`-delimited
   outputs of the step-native SFT (15.8 fine steps per solution); these are natural CoT
   trajectories retro-segmented by the old segmenter (10.6 coarser steps), from problems of
   levels 1–3. So the §5.3 conclusion sharpens: the AUC gain is specific to the step-native
   generator *and* the MATH-train difficulty mix, and it does not survive either shift.
2. **No policy selects for formalization on this cache.** Every pick − random is within
   ±0.05 with a CI straddling zero; the random → oracle gap (0.606 → 0.806) is untouched.
   Within-problem ρ is ≈ 0 for the forked classifier and +0.23 for the old verifiability
   one, which now also prefers *shorter* trajectories less than §10 reported (ρ with step
   count −0.20 vs −0.29 for the cached Qwen3 scores, +0.01 for forked).
3. **The forked pick's higher answer accuracy (0.867 vs 0.753, +1.7 problems) is not
   significant** (CI [−0.03, +0.27]) and is not mediated by formalization: on this cache
   picking the trajectory with the *true* highest Lean rate gives accuracy 0.733, i.e. no
   better than random (0.753), unlike the +7 points of §5.4 on the harder MATH-train trees.
   Correct trajectories do have a higher true rate pooled across problems (0.687 vs 0.361),
   but within a problem only 7 of 15 vary in both, ρ = +0.25 on those.
4. **The relabelling itself changes a third of the verdicts** (*the* caveat for reading
   this table against §10 / §10.2): on the 1,586 steps labelled under both renderings the
   per-step agreement is **66.6%** — 213 steps proved only under the legacy rendering,
   317 only under `facts`; 52 of the 84 legacy formalize-stage failures (the `_THEOREM_RE`
   binder bug, §7.2) now prove. Overall proved rate 0.510 → 0.576. Since the legacy
   rendering formalized the *problem* for 33–38% of steps (`experiments_v1.md` §9.2), the
   §10 pick statistics — including "worse than random" — were partly measured against the
   wrong claims; under step-level labels every old classifier's pick gain is ≥ 0 but within
   noise. §10's qualitative conclusion (no usable selection signal from any classifier on
   this cache) stands under both label sets.

Bottom line for the programme: on three evaluation sets — held-out forked trees (§7.4),
step-native MATH-500 decodes (§5.3) and this natural-CoT cache — the retrained classifier
beats the FaithformBench ones as a step judge only on the first, in-distribution one, and
never turns into a selection gain. The next lever is not another classifier fit on the
same 150 trees but either (a) training data from the generator and segmentation actually
decoded (natural CoT + retro-segmentation, or step-native for both), at a size that
supports the ranking losses, or (b) Lean in the loop (`rejection` / hybrid), which does
not depend on the judge's out-of-distribution AUC.

## 5. Experiment 2 — constrained beam search with a boundary-trained classifier

Not started; **its training data is already collected by §3** (the sibling pairs
with their Lean verdicts *and* each sibling's continuation outcome).

Design:

1. **Training unit** = one fork: `(prefix, step_a, step_b)` with labels
   `proved_a, proved_b` and continuation targets `true_rate_a, true_rate_b`,
   `correct_a, correct_b` (`ForkedTree.sibling_pairs`, `build_sibling_pairs(target=
   "proved" | "continuation")`). The classifier scores `render(question, prefix,
   step)` exactly as `lean_constrained_beam_search`'s classifier intervention does
   (`predict_step_logit_for_beam`), so train and decode see byte-identical inputs.
2. **Loss**: pairwise on the proved target (primary — the training target does
   not change, per the standing constraint) with the continuation rate as an
   auxiliary head/loss (a value estimate the beam can use for pruning); listwise
   once n>2 siblings per boundary exist.
3. **Second pass that would strengthen it** (not needed to start): at each fork
   point sample n=4 siblings instead of 2 and continue each — 3 extra
   continuations per boundary, i.e. roughly the cost of §3 again — giving
   listwise groups at every boundary.
4. **Decode**: `lean_constrained_beam_search` (`INTERVENE_CLASSIFIER`, then
   `HYBRID`) with the step-native Qwen2.5-7B and the boundary-trained classifier,
   `tau_phi` fitted by `fit_beam_calibration` on the held-out sibling pairs rather
   than on FaithformBench; evaluated on the 110-problem set against
   `base_greedy` / `self_consistency` / rerank as in `experiments_v1.md` §9.
5. **Why this can work where §10 didn't**: the classifier is trained on the exact
   decision it is asked to make (which of *these* continuations of *this* prefix,
   from *this* generator, proves), so neither the difficulty confound nor the
   generator shift applies.

### 5.1 As run (Oct 6)

`scripts/exp2_beam_decode.py` (in `/scratch/users/iacopog/rllf-runs/forked_tree/scripts/`), the
110-problem MATH-500 set (first 110 of `ff_qwen7b/math500_base_greedy.jsonl`), step-native
Qwen2.5-7B, `INTERVENE_CLASSIFIER`, four arms on the same generator:

| arm | beam | expansion | α | classifier |
|---|---|---|---|---|
| `greedy` | 1 | 1 | 0 | — |
| `beam_lm` | 4 | 2 | 0 | — (the search alone: the control every classifier arm is read against) |
| `beam_forked` | 4 | 2 | 1 | `forked_supervised` (§7.4) |
| `beam_oldverif` | 4 | 2 | 1 | old FaithformBench `prm_verifiability_all8_majority` |

Deviations from the design above, both decided on §7.4's results / a dry run:

- **Classifier = `forked_supervised` (BCE), not a pairwise-trained one.** Pairwise collapsed at
  full data (step AUC 0.48, memorised by epoch 1); BCE on the same `(question, prefix, step)`
  states is the one that learned the step decision (AUC 0.69). Same rendering as the beam.
- **`tau_phi`** fitted on the 980 held-out tree steps (also the set used for epoch selection and
  `step_calibrator`; it only rescales). Fitted: forked T=2.03, b=0.68; old verifiability T=6.27,
  b=6.79.
- **Mean, not sum, over a beam's classifier scores** (`BeamSearchConfig.classifier_aggregate`,
  new; default stays `"sum"`). The CPU dry run gave forked calibrated logits of about +0.8 to +1.0
  per step against an LM score of about −0.2 per step: summed, an unfinished beam outscores every
  finished one (whose score is frozen), so the search never chooses to stop and runs to the
  200-step ceiling; the old classifier's negative logits would instead favour the shortest
  solution. Either way the classifier's sign, not its ranking, would decide trajectory length.
  Covered by `tests/test_beam_search.py::test_classifier_aggregate_controls_the_length_preference`.

Hybrid (real-Lean escalation below a confidence threshold) is not run: with 28% of steps
proved, most steps fall below 0.5 and would be escalated — ~100 Lean calls per problem, ~300
GPU-h for 110 problems.

### 5.2 Decode results (Oct 7) — accuracy only, formalization not yet measured

4.4 GPU-h (one job; greedy 12 min, each beam arm 68–93 min). Outputs
`/scratch/users/iacopog/rllf-runs/forked_tree/exp2_beam/<arm>.jsonl`; analysis `exp2_analyse.py`.
110 problems, levels 1/2/3 = 22/45/43. Paired exact McNemar (wins/losses):

| arm | acc | L1 | L2 | L3 | steps | words/step | words/sol. | final ≠ `beam_lm` | vs `beam_lm` | vs `greedy` |
|---|---|---|---|---|---|---|---|---|---|---|
| `greedy` | 0.591 | 0.636 | 0.644 | 0.512 | 8.9 | 9.9 | 88 | 105/110 | 7/14, p=0.19 | — |
| `beam_lm` | 0.655 | 0.727 | 0.756 | 0.512 | 9.3 | 11.6 | 108 | — | — | 14/7, p=0.19 |
| `beam_forked` | **0.673** | 0.773 | 0.711 | 0.581 | 10.8 | 9.2 | 100 | 73/110 | 9/7, p=0.80 | 20/11, p=0.15 |
| `beam_oldverif` | 0.645 | 0.636 | 0.667 | 0.628 | 12.2 | 12.1 | 148 | 69/110 | 9/10, p=1.00 | 16/10, p=0.33 |

- **The classifier does steer the search** — it changes the chosen trajectory on 66% (forked) /
  63% (old) of problems — **but does not move accuracy**: every difference to the search-only
  control is a few problems either way, nowhere near significance. Beam search itself is worth
  ~+6 points over greedy (p=0.19). Accuracy was never the classifier's training target; the
  experiment's own question is formalization (below).
- **The two classifiers steer in different directions.** Forked splits solutions into finer
  steps (9.2 vs 11.6 words/step, similar total length); the old verifiability classifier makes
  solutions 37% longer. The first matters for the next measurement: a classifier that prefers
  short, concrete steps can raise a trajectory's *mean* proved rate by splitting alone, so the
  real-Lean comparison must be read alongside granularity (e.g. proved *content* per solution,
  not only the step rate).
- Step-native greedy (0.591) is below the non-step-native Qwen2.5-7B greedy on the same problems
  (0.664 after the §9.1 re-grade, `experiments_v1.md`); the SFT on 226 solutions costs some
  accuracy.

Real-Lean labelling cost (unique steps, prefixes shared across arms deduplicated): `beam_lm` +
`beam_forked` 1,874 (~55 GPU-h at the forked-tree job's ~35 steps/GPU-h); + `beam_oldverif`
2,759 (~80); all four 3,580 (~100).

### 5.3 Real-Lean result (Oct 9) — no significant formalization gain

`exp2_real_lean.py`: 1,874 unique steps of `beam_lm` + `beam_forked`, same pipeline/settings
as §7.2 (facts rendering, 8 prover samples, 60 s, maxHeartbeats 400000); 21.2 GPU-h (89
steps/GPU-h on an uncontended card); 44.8% proved (MATH-500 levels 1–3, easier than the
MATH-train 1–5 trees' 27.6%). Labels `exp2_beam/real_lean.json`; analysis
`exp2_lean_analyse.py`. All 110 problems fully labelled in both arms; paired per problem:

| measure | `beam_lm` | `beam_forked` | forked better / worse | sign p | Wilcoxon p |
|---|---|---|---|---|---|
| step rate (proved / steps) | 0.526 | 0.552 | 39 / 28 | 0.22 | 0.22 |
| word share (words in proved steps) | 0.522 | 0.534 | 37 / 33 | 0.72 | 0.54 |
| proved steps per solution | 4.26 | 5.06 | 36 / 26 | 0.25 | 0.27 |
| correct | 0.655 | 0.673 | 9 / 7 | 0.80 | 0.62 |

By level the direction is the same (step rate L1 0.519 → 0.585, L2 0.556 → 0.560, L3 0.499 →
0.527), nothing significant; the granularity-robust word share moves by +1.2 points overall
and is lower in L2.

**Why — the classifier does not transfer to new problems.** Step AUC on these 1,874 MATH-500
steps: `forked_supervised` **0.630**, old FaithformBench verifiability **0.626**. On the
held-out forked trees (same MATH-train distribution as its training data) it was 0.692 vs
0.599. So the gain of §7.4 is largely in-distribution (problem source / difficulty), not
general to the generator's steps; on new problems the new classifier is no better a step
judge than the old one, and a ~0.63-AUC judge steering 8 candidates per round moves
formalization by a couple of points at most. (These steps are partly the forked
classifier's own picks, which if anything should favour it.)

Reading of experiment 2: the search + calibration + mean aggregation machinery works (the
classifier changes two-thirds of the chosen solutions, without runaway or length collapse),
and the result is a clean null on formalization and accuracy. The bottleneck is the step
judge's out-of-distribution AUC, not the search.

### 5.4 Ceiling check — does the *true* Lean rate predict correctness? (Oct 9, CPU)

On the 150 forked trees (4 trajectories each, all labelled): picking each problem's trajectory by
its **real** Lean-proved step rate gives answer accuracy **0.386** vs random **0.318** (any-correct
oracle 0.547); within-problem Spearman ρ(true rate, correct) = **+0.25** (57 problems with variation
in both). So formalization does carry answer signal — a perfect step judge would be worth ~+7
points of accuracy in reranking on this data (single split, no significance test). The
experiments above fail to get that because the learned judge reaches only AUC ~0.63 on new
problems (§5.3), not because the target is useless.

## 6. Files

| file | role |
|---|---|
| `rllf/data/forked_tree.py` | `ForkedTree`/`ForkedTrajectory`, `step_key`, `unique_steps`, `sibling_pairs`, label→example builders (`build_prm_examples`, `build_sibling_pairs`, `build_problem_pairs`, `build_trajectory_groups`), `split_trees`, `summarize`, JSONL I/O |
| `rllf/generation/forked_sampling.py` | `ForkConfig`, `continue_trajectory`, `sample_forked_tree` |
| `rllf/generation/base_model.py` | `StepNativeHFBaseModel` (SFT prompt, special tokens preserved) |
| `rllf/data/step_sft.py` | `render_step_sft_prompt` (shared by trainer and decoder) |
| `rllf/policy/training.py` | `train_trajectory_listwise`, `evaluate_trajectory_groups` |
| `rllf/evaluation/forked_tree_eval.py` | held-out tree metrics, `format_tree_report` |
| `rllf/protocol.py` | `extract_boxed` (balanced braces) |
| `scripts/train_forked_classifier.py` | experiment 1 fits + comparison table |
| `tests/test_forked_tree.py` | 12 tests (sampler on a scripted step model, dedup, pairs, builders, listwise training, evaluation) |
| scratchpad `forked_tree_sample.py` / `forked_tree_real_lean.py` (+ `launch_*.sh`) | the two GPU jobs; outputs under `/scratch/users/iacopog/rllf-runs/forked_tree/` |
| `…/forked_tree/scripts/launch_qwen3_refit.sh`, `exp2_step_auc.py`, `rerank15_bootstrap.py` | §8: Qwen3-1.7B refit chain, step AUC per evaluation set, rerank bootstrap |
| `…/forked_tree/scripts/beam_decode_v2.py`, `negation_probe.py`, `binder_rerun.py`, `hybrid_beam_decode.py`, `hard_forked_sample.py`, `regrade_arms.py` | §8.6 α sweep / hard set, §9 refutation probe, `experiments_v1.md` §9.4 binder re-run, Lean-in-the-loop, hard forked trees, answer re-grade |

## 7. Results

### 7.1 Sampling (done — 1.40 GPU-h, 1 h 24 min, 33 s/problem average)

| | |
|---|---|
| problems / trajectories | 150 / 600 (30 per level) |
| steps total / unique | 9,460 / **5,415** — a 43% Lean saving from shared prefixes (estimate was 35%) |
| steps per trajectory | 15.8 (the step-native model writes finer steps than natural Qwen2.5-7B's 10.6 after retro-segmentation) |
| sibling pairs (distinct steps behind an identical prefix) | 320 of 450 forks (the rest: identical re-draw, or the fork produced `<final>` where the parent had a step) |
| reached a final answer | 589 / 600; every stop reason was `final`, none hit the safety ceiling |
| correct (after re-grade) | 191 / 600 = 32% (79 before `MathEquivalenceChecker`); MATH-train levels 4–5 are much harder than the MATH-500 subset the 66% figure comes from |

Lean budget at 5,415 unique steps ≈ 83 GPU-h at the observed 65 steps/GPU-h.
`forked-tree-lean` is queued (48 h limit, resumes from its every-3-steps checkpoint;
work is ordered round-robin over levels so any partial label file is level-balanced).

### 7.2 Lean labels — done (5,415 / 5,415, 27.6% proved)

The first 2.5 h of labelling (198 steps, 87 steps/GPU-h) surfaced a bug in the
pipeline itself: `_THEOREM_RE` required a `:` straight after the theorem name, so
every Goedel statement with explicit binders — `theorem step_0 (A B C : ℝ)
(h : A + B + C = π) : …`, i.e. any step that mentions variables or hypotheses — was
rejected as "no parseable theorem statement" and counted as a formalize-stage
failure. On the triangle-angle problem that was 127/127 steps; binder-free concrete
arithmetic passed. Fixed (`rllf/verification/real_lean_pipeline.py`, tested in
`tests/test_real_lean_pipeline.py`), the 127 wrongly-failed entries were dropped from
the cache, and the job was resubmitted (the 71 prove/proved verdicts are kept).

This bug predates this run: the earlier real-Lean caches carry 84 (§10), 79/74
(§8.1/§8.2 arms), 154 (§4) and 216 (§8.3) formalize-stage failures — 10–17% of their
steps — a large share of which are binder statements that were never sent to the
prover. Since those caches store the failing step, re-running *only* those ~607
entries with the fix costs ~7 GPU-h and is queued as an open item; until then every
earlier real-formalization rate is a lower bound biased against symbolic/contextual
steps.

**Second restart — the pipeline was formalizing the problem, not the step.** After the
regex fix the proved rate stayed at 3% and throughput fell to 20 steps/GPU-h. The stored
statements showed why: "Definition: cot θ = cos θ/sin θ" was formalized as
`Real.cot (−π/3) = −√3/3` (the problem's answer), "In any triangle A+B+C=π" as the whole
determinant, "Let θ₁ = π/70" as the entire reflection problem. `render_step_claim` leads
with `Problem: …` and Goedel's prompt asks for "the following problem statement". This is
not specific to the step-native model: the older caches that stored statements show
33–38% of steps sharing a *verbatim* conclusion with another step of the same problem
(`experiments_v1.md` §9.2). Job killed at 371 labels; all of them discarded.

Formalize-only probe on 37 steps from 5 trees (`claim_rendering_probe*.py`, ~5 GPU-min
each), scoring each rendering by token overlap of the Lean statement with the step vs.
with problem-only tokens, and by how often the statement equals Goedel's formalization
of the bare problem:

| rendering | step overlap | problem-only overlap | notes |
|---|---|---|---|
| current (`Problem / Previous steps / Step to verify`) | 0.471 | 0.258 | mostly the problem or its answer |
| `prove_that` ("In the context of …, assuming (1)…(n), prove that {step}") | 0.599 | 0.277 | tracks the step but smuggles the answer back in (`… ∧ m = 189`) |
| `claim_last` ("Setting (background only) / Facts / The statement to formalize is exactly: Claim: …") | 0.512 | **0.210** | clean step statements: `A + B + C = π`, `sin(π/3) = √3/2`, `∀θ, sin(−θ) = −sin θ`; still falls back to the problem on under-specified steps |

Probe v2, same 37 steps:

| rendering | step overlap | problem-only overlap | notes |
|---|---|---|---|
| `claim_last` (repeat) | 0.488 | 0.208 | still the determinant on 5 of the 8 cofactor/minor steps |
| `claim_first_last` (claim stated before *and* after the background) | 0.552 | 0.146 | better, but still the matrix on 3 of 8 |
| **`facts_only`** (no problem text; previous steps as numbered hypotheses; `Statement to formalize: {step}`) | **0.606** | **0.158** | the conclusions *are* the steps: `(a13 = 1 ∧ a33 = 1)`, `(−1)^(2+4)·1 = 1`, `cofactor_14 = −1`, `R (R l) = l`, `θ₁ = π/70 ∧ θ₂ = π/54`, `m_refl = (m·tanθ − tanθ)/(1 − m·tan²θ)` |

Every rendering that kept the problem text, however de-emphasised, fell back to it on
some steps; removing it is what works. **Adopted:** `render_step_claim(…, style="facts")`
is now the default in `rllf/verification/real_lean_pipeline.py` (`style="legacy"`
reproduces the old prompt); `RealFormalizabilityScorer(claim_style=…)` threads it
through. Known trade-off: a step whose variables are introduced only in the problem
statement and never restated gets formalized without that context — in the probe the
generators restated them in the previous steps every time, and a definition step
(`Let θ₁ = π/70`) is now trivially provable, which is the same property FaithformBench's
own step labels have. Labelling relaunched from zero under the new rendering
(`real_lean_legacy_rendering_discarded.json` keeps the 371 old verdicts for reference).

**Throughput and the label settings (decided at label #1026).** Under the step-level
rendering the job settled at ~20 steps/GPU-h across five trees (≈270 GPU-h for the full
set) with 28% proved (79% on the level-1 tree, 12–25% on levels 2–5). The log showed
where the time went: prover candidates parse cleanly (0% runaway, 13.8/16 spliced) and
successes come early — the first successful candidate is within the first 4 for 73% of
proved steps and within the first 8 for 86% — while every candidate was Lean-checked
with VeriForm's default **300 s timeout and `maxHeartbeats 0`**, so a failing step (72% of
them) paid the slowest candidate's full search. Adopted (user decision): **8 prover
samples, Lean timeout 60 s, `maxHeartbeats 400000`** (`RealFormalizabilityScorer(
prover_batch_size=8, lean_timeout=60, max_heartbeats=…)`, `cap_heartbeats` rewrites
Goedel's header). Expected cost: ~14% fewer positives than n=16 plus rare timeouts, for a
3–4× speed-up (~70–90 GPU-h for the remainder). The first 1,026 labels (n=16, 300 s,
unbounded) are kept and are at least as lenient; every label now records its `settings`.

### 7.3 Experiment 1 fits — round 1 at the halfway mark (2,700 labels, snapshot)

MiniLM-L6, by-problem 120/30 tree split, CPU. Fit side: 2,439 labelled states (29%
proved), 254 sibling pairs of which 43 labelled on both sides, 62 complete trajectory
groups. **Held-out side is still small — 261 labelled steps, 9 complete groups, 6
labelled sibling pairs — so every number below is direction, not result.**

| policy | step AUC vs Lean | PRM pick | ORM pick | random | oracle | PRM ρ | ORM ρ |
|---|---|---|---|---|---|---|---|
| `forked_supervised` (BCE, best epoch = 2) | **0.643** | 0.465 | 0.482 | 0.418 | 0.516 | **+0.297** | **+0.334** |
| `forked_pairwise` (last epoch) | 0.531 | 0.396 | 0.439 | 0.418 | 0.516 | −0.230 | +0.243 |
| `forked_listwise` (last epoch) | 0.394 | 0.417 | 0.446 | 0.418 | 0.516 | −0.221 | +0.248 |
| `forked_orm_true_rate` (outcome head only) | 0.373 | 0.444 | 0.412 | 0.418 | 0.516 | +0.119 | −0.018 |
| old `prm_verifiability_all8_majority` | 0.560 | 0.451 | 0.464 | 0.418 | 0.516 | −0.041 | +0.117 |
| old `prm_agreement_all8_majority` | 0.614 | 0.497 | 0.410 | 0.418 | 0.516 | +0.343 | −0.290 |
| old `prm_verifiability_gpt-5.2` | 0.463 | 0.427 | 0.416 | 0.418 | 0.516 | +0.074 | −0.098 |

Reading: (i) the same-problem data with the *old* loss (BCE, held-out-selected epoch)
already turns the within-problem correlation positive (+0.30 PRM / +0.33 ORM vs −0.04 for
the old verifiability policy) and gives the best step AUC — the thing the redesign was
for; (ii) the two ranking objectives are not informative in this round for tuning
reasons, not data reasons: pairwise memorised (train loss 0.79 → 0.06 while held-out pair
accuracy peaked at epoch 0–1 and decayed) and the reported model is the last epoch;
listwise barely trained (loss 2.21 → 2.18 at lr 3e-4); ORM had 62 groups and stayed at the
uniform-softmax loss (ln 4 ≈ 1.386). Round 2 (queued on the same snapshot): best-held-out-
epoch restore for the ranking trainers (`_BestCheckpoint`, `--patience`), listwise at
lr 1e-3, and BCE + auxiliary pair loss (`--rank-weight 0.5`). (iii) The old agreement
classifier again transfers better than the old verifiability one (0.614 vs 0.560; ρ +0.34)
— third time this direction has appeared, still on a small held-out set.

**Round 2 (same snapshot, best held-out epoch restored):**

| policy | step AUC | sibling acc | PRM pick | ORM pick | PRM ρ | ORM ρ | best epoch |
|---|---|---|---|---|---|---|---|
| `forked_pairwise_r2` | 0.642 | 0.583 | 0.437 | 0.389 | +0.050 | −0.418 | 0 (pair acc 0.597) |
| `forked_listwise_r2` (lr 1e-3) | 0.607 | 0.667 | 0.421 | 0.411 | −0.006 | −0.005 | 2 (pair acc 0.470) |
| `forked_supervised_rank0.5_r2` (BCE + 0.5·pair) | 0.642 | 0.750 | 0.464 | 0.428 | +0.198 | +0.043 | 0–1 |
| `forked_supervised` (round 1, BCE) | 0.643 | 0.583 | 0.465 | 0.482 | +0.297 | +0.334 | 2 |

*Answer accuracy of the picked trajectory* (same 9 held-out groups; random pick 0.278,
oracle — any correct trajectory — 0.444, so one problem = 0.111): PRM / ORM pick —
`forked_supervised` 0.333 / **0.444**, `+rank0.5` 0.444 / 0.333, `pairwise_r2` 0.444 / 0.444,
`listwise_r2` 0.222 / 0.333, `orm` 0.333 / 0.222, old verifiability 0.222 / 0.444, old
agreement 0.444 / 0.111. Too coarse to rank policies at this size; the metric is now part
of the standard report (`format_tree_report`, second block) for the full-data run.

With early stopping the ranking objectives reach the same step AUC as BCE (0.64) but add
nothing to within-problem selection (ρ ≈ 0 to +0.2 vs +0.30 for plain BCE), and the
held-out pair accuracy they optimise never exceeds 0.60. On this snapshot plain BCE on
the same-problem data is the best of the new fits. Held-out side still 9 groups / 6
sibling pairs — the ranking conclusion in particular should be re-read at full data,
since the ranking losses are the ones that need many labelled sibling pairs (43 on the
fit side so far).

Smoke run of `scripts/train_forked_classifier.py --objective pairwise` on the first 1,326
labels (CPU, 5.6 min): 120/30 tree split, 14 sibling + 478 within-problem + 24 same-prefix
pairs, held-out pair accuracy 0.63 → 0.74 over two epochs, comparison table and
checkpoint written. **Not results** — the held-out side had 128 labelled steps, 5 complete
trajectory groups and 3 labelled sibling pairs. One early signal to re-check at scale: on
those steps the old *agreement* classifier scores step AUC 0.688 and the old verifiability
one 0.634 against the step-level labels, both above the 0.55–0.61 they scored against the
problem-level labels in `experiments_v1.md` §10.2.

**Completion (Oct 1–3).** Two more interruptions, neither touching the labels: (i) the
job died with the session that had launched it (`gpuq submit &` was a child of that
shell; exit −15 at 4,770) — relaunched detached (`setsid nohup`), with the conda env
pinned in the launch script and the scripts moved to
`/scratch/users/iacopog/rllf-runs/forked_tree/scripts/`; (ii) at 5,295 one step's
formalizer prompt was 8,267 tokens, over Goedel-Formalizer's `max_model_len=8192` — the
`facts` rendering lists every previous step, so deep steps of long trajectories grow.
Fixed by `max_model_len=16384` plus a guard that records a prompt that still does not fit
as a formalize failure instead of crashing (prompt and sampling unchanged, so earlier
labels stay comparable). Final: 5,415 labels, 27.6% proved; 138/150 complete trajectory
groups; 98 sibling pairs labelled on both sides (254 with continuation targets).

### 7.4 Experiment 1 fits — full label set

`run_forked_fits_full.sh`: MiniLM-L6, same by-problem 120/30 split (`seed 0`), CPU, best
held-out epoch restored for every objective. Fit side 4,435 labelled states, 1,886
within-problem pairs (73 sibling), 109 trajectory groups; held-out 980 steps, 29–30
groups. Outputs: `/scratch/users/iacopog/rllf-runs/forked_tree/classifiers_full/`, log
`logs/forked_fits_full.log`.

*Formalization of the picked trajectory* (true Lean-proved step rate; random 0.302,
oracle 0.422):

| policy | step AUC | sibling acc | PRM pick | ORM pick | PRM ρ | ORM ρ | training note |
|---|---|---|---|---|---|---|---|
| `forked_supervised` (BCE) | **0.692** | 0.500 | **0.344** | 0.275 | +0.158 | −0.108 | AUC rises every epoch 0.660 → 0.692 |
| `forked_supervised_rank0.5` | 0.685 | 0.500 | 0.336 | 0.332 | +0.157 | +0.202 | best epoch 1 |
| `forked_pairwise` | 0.480 | 0.500 | 0.302 | 0.285 | +0.008 | −0.062 | best epoch 0 (pair acc 0.594), then memorises |
| `forked_listwise` (lr 1e-3) | 0.524 | 0.480 | 0.290 | 0.316 | +0.047 | +0.577 | loss flat at 2.23 — did not train |
| `forked_orm_true_rate` | 0.383 | 0.540 | 0.302 | 0.308 | −0.036 | +0.166 | loss ≈ ln 4 — did not train |
| `forked_orm_correct` | 0.362 | 0.620 | 0.283 | 0.320 | −0.122 | +0.100 | loss ≈ ln 4 — did not train |
| old `prm_verifiability_all8_majority` | 0.599 | 0.540 | 0.323 | 0.297 | +0.068 | +0.003 | |
| old `prm_agreement_all8_majority` | 0.478 | 0.460 | 0.341 | 0.334 | +0.173 | −0.005 | |
| old `prm_verifiability_gpt-5.2` | 0.552 | 0.500 | 0.321 | 0.292 | +0.118 | −0.033 | |

*Answer accuracy of the picked trajectory* (random 0.275, oracle 0.467; one problem =
0.033), PRM / ORM: `forked_supervised` 0.333 / 0.233, `+rank0.5` 0.300 / 0.333,
`pairwise` 0.267 / 0.167, `listwise` 0.367 / 0.267, `orm_true_rate` 0.200 / 0.400,
`orm_correct` 0.200 / 0.333; old verifiability 0.300 / 0.367, old agreement
0.333 / 0.200, old gpt-5.2 0.333 / 0.200.

Reading:

1. **The distribution problem is fixed; the selection problem only partly.** Plain BCE
   on the generator's own Lean-labelled steps gives step AUC 0.692 on held-out problems,
   against 0.48–0.60 for every FaithformBench-trained classifier on the same steps (and
   against the 0.50–0.61 of §10.2). That gain does not carry over to choosing between a
   problem's trajectories: PRM pick 0.344 recovers about a third of the random → oracle
   gap (0.302 → 0.422), within-problem ρ is +0.16, and the old agreement classifier
   reaches the same (0.341, +0.17). Between-policy differences in pick rate and accuracy
   are 1–2 problems out of 30 — not separable at this size.
2. **The halfway signal shrank.** The +0.30 / +0.33 ρ of §7.3 (9 held-out groups) is
   +0.16 / −0.11 on 30 groups; the halfway number was mostly noise.
3. **No ranking objective helps.** Pairwise peaks at epoch 0 and then memorises the 73
   sibling pairs; listwise and both ORMs never leave the uniform-softmax loss. The ORM
   targets differ within a group by little (sibling continuations share most of their
   steps), so a single trajectory-level score per solution gives a near-zero gradient.
   Listwise's ORM ρ +0.577 sits with a 0.316 ORM pick (≈ random) and an untrained loss;
   treat it as noise, not signal.
4. **What this means for the plug-in.** `forked_supervised` is the checkpoint to try in
   `full_trajectory_rerank`, but the held-out trees predict at best a small gain over
   random selection there. Step-level discrimination (experiment 2's boundary decision,
   where the classifier is asked about one step behind a fixed prefix) is where the AUC
   gain is most directly usable; the sibling accuracy of 0.50 above is on only 25
   held-out pairs labelled on both sides.

## 8. Capacity: refitting the forked classifier with a Qwen3-1.7B encoder (Oct 14–15)

Question: is the OOD step-AUC ceiling of §5.3 / §4.1 (0.57–0.63) a *capacity* limit of
MiniLM-L6, as `experiments_v1.md` §7 found offline for FaithformBench (0.798 → 0.863 with a
fine-tuned Qwen3-1.7B)? Same data (150 trees, 5,415 labels, by-problem 120/30 split, seed
0), same `supervised` BCE objective, encoder `Qwen/Qwen3-1.7B` with the §7 recipe
(last-token pooling, lr 2e-5, batch 16, max length 320, 5 epochs, patience 2, best
held-out epoch restored). `launch_qwen3_refit.sh` chains fit → beam decode → Lean labels;
8.9 GPU-h in one job (fit 30 min, decode 60 min, Lean 673 new unique steps at 92
steps/GPU-h). Checkpoint `classifiers_full/forked_supervised_qwen3`; log
`logs/qwen3_refit.log`.

### 8.1 Held-out forked trees (30 problems)

| policy | step AUC | sibling acc | PRM pick | ORM pick | random | oracle | PRM ρ | ORM ρ | PRM acc | random acc |
|---|---|---|---|---|---|---|---|---|---|---|
| `forked_supervised_qwen3` | **0.753** | **0.680** | **0.355** | 0.301 | 0.302 | 0.422 | **+0.220** | +0.129 | 0.267 | 0.275 |
| `forked_supervised` (MiniLM, §7.4) | 0.692 | 0.500 | 0.344 | 0.275 | 0.302 | 0.422 | +0.158 | −0.108 | 0.333 | 0.275 |
| old `prm_verifiability_all8_majority` | 0.599 | 0.540 | 0.323 | 0.297 | 0.302 | 0.422 | +0.068 | +0.003 | 0.300 | 0.275 |
| old `prm_agreement_all8_majority` | 0.478 | 0.460 | 0.341 | 0.334 | 0.302 | 0.422 | +0.173 | −0.005 | 0.333 | 0.275 |

Held-out AUC by epoch 0.715 → 0.725 → 0.746 → 0.741 → 0.753 (still rising at epoch 4;
a longer schedule may add a little). Capacity does matter for the step decision: +0.06 AUC
and sibling-pair accuracy 0.50 → 0.68 (on the 25 pairs labelled on both sides). It barely
matters for trajectory selection: PRM pick +0.011 over MiniLM (one third of a problem's
worth), answer accuracy of the pick at random level.

### 8.2 Step AUC against real Lean on every evaluation set

| evaluation set (generator / segmentation / problems) | n steps | MiniLM `forked_supervised` | **Qwen3 `forked_supervised_qwen3`** | old verifiability |
|---|---|---|---|---|
| held-out forked trees (step-native / MATH-train 1–5) | 980 | 0.692 | **0.753** | 0.599 |
| exp-2 beam arms, all labelled (step-native / MATH-500 1–3) | 2,547 | 0.631 | **0.720** | 0.642 |
| – of which `beam_lm` steps (no classifier in the loop) | 1,023 | 0.605 | **0.722** | 0.579 |
| – `beam_forked` (MiniLM's own picks) | 1,189 | 0.643 | **0.736** | 0.686 |
| – `beam_forked_qwen3` (its own picks) | 1,021 | 0.600 | 0.667 | 0.663 |
| §10 rerank cache (natural CoT, retro-segmented / MATH-500 1–3) | 1,556 | 0.572 | **0.654** | 0.599 |

(`exp2_step_auc.py`, `rerank15_analyse.py`.) The in-distribution gain of §7.4 now
*transfers*: on the generator's MATH-500 steps the Qwen3 classifier is +0.09 over MiniLM
and +0.08 over the old classifier, and it is the first retrained checkpoint that beats
the old one on the natural-CoT cache (0.654 vs 0.599). The lower AUC on its own picks
(0.667) is the usual selection effect — those steps are the ones it already rated high.
So the §5.3 / §4.1 ceiling was a capacity limit, not a data limit.

### 8.3 Constrained beam search with it (110 problems, same settings as §5.1)

`beam_forked_qwen3`: beam 4, expansion 2, α = 1, mean aggregation, `tau_phi` fitted on
the 980 held-out tree steps (T = 8.30, b = 5.10 — a much flatter calibrated scale than
MiniLM's T = 2.03). 60 min decode. Paired against the search-only control and the MiniLM
arm (`exp2_analyse.py`, `exp2_lean_analyse.py`); all 110 problems fully Lean-labelled in
all three arms (2,547 unique steps, 46.1% proved):

| arm | acc | L1 | L2 | L3 | steps/sol | words/step | words/sol | final ≠ `beam_lm` | acc vs `beam_lm` (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| `beam_lm` | 0.655 | 0.727 | 0.756 | 0.512 | 9.3 | 11.6 | 108 | — | — |
| `beam_forked` (MiniLM) | 0.673 | 0.773 | 0.711 | 0.581 | 10.8 | 9.2 | 100 | 73/110 | 9/7, p = 0.80 |
| `beam_forked_qwen3` | 0.627 | 0.727 | 0.733 | 0.465 | 9.3 | 9.0 | 84 | 88/110 | 8/11, p = 0.65 |

| formalization, paired per problem | `beam_lm` | `beam_forked_qwen3` | better / worse | sign p | Wilcoxon p |
|---|---|---|---|---|---|
| step rate (proved / steps) | 0.526 | 0.557 | 43 / 33 | 0.30 | 0.08 |
| word share (words in proved steps) | 0.522 | 0.543 | 43 / 40 | 0.83 | 0.22 |
| proved steps per solution | 4.26 | 4.76 | 44 / 24 | **0.021** | **0.014** |
| correct | 0.655 | 0.627 | 8 / 11 | 0.65 | 0.49 |

vs `beam_forked` (MiniLM): step rate 0.552 → 0.557 (40/35, p = 0.64), word share 0.534 →
0.543 (p = 0.44), proved steps 5.06 → 4.76 (p = 0.63), correct 0.673 → 0.627 (10/15,
p = 0.42) — indistinguishable. By level vs `beam_lm` the step-rate gain is L1 0.519 →
0.589, L2 0.556 → 0.559, L3 0.499 → 0.539; the only significant cell is level-1 proved
steps (8/1, p = 0.04).

Reading:

1. **A judge at AUC 0.72 on the generator's own steps still does not move the search's
   outcome.** It steers more than MiniLM did (changes the final answer on 80% of problems
   vs 66%), and it shifts every formalization measure in the right direction, but only the
   count of proved steps is significant, and that one is granularity-sensitive; the
   word-share measure (+2 points, p = 0.22) is the honest estimate. Accuracy is 3 points
   *below* the control (−3 problems, well inside noise), consistent with §5.4's finding
   that on these easy problems the Lean rate carries little answer signal.
2. **The classifier makes solutions shorter (84 words vs 108), not finer.** Unlike the
   MiniLM arm (more, shorter steps) it keeps 9.3 steps and shortens each one. A judge that
   rewards short concrete steps compresses the write-up; whether that is "more
   formalizable reasoning" or "less reasoning per step" is what word share vs step rate
   is for, and word share says the gain is small.
3. **Why so little, given the AUC jump?** With α = 1 and mean aggregation the calibrated
   classifier term is divided by T = 8.3, so its per-step contribution is small against
   the LM log-probabilities; and the decision is over only 8 candidates per round, most of
   which share a prefix, so there is little for a judge to separate. Under the same
   calibration-by-held-out-AUC recipe, more judge quality translates into a *flatter*
   calibration, not a stronger push. An α sweep (and/or fitting `tau_phi` on the
   sibling pairs, as §5 originally specified) is the obvious untested knob.

### 8.4 `full_trajectory_rerank` on the re-labelled 15-problem cache

Same protocol as §4.1 (`rerank15_analyse.py`, `rerank15_bootstrap.py`):

| policy | step AUC | pick | random | oracle | pick − random [95% CI] | pick acc | random acc | within-problem ρ |
|---|---|---|---|---|---|---|---|---|
| `forked_supervised_qwen3` | **0.654** | **0.683** | 0.606 | 0.806 | **+0.076 [+0.013, +0.138]** | 0.667 | 0.753 | +0.140 |
| `forked_supervised` (MiniLM) | 0.572 | 0.622 | 0.606 | 0.806 | +0.015 [−0.079, +0.106] | 0.867 | 0.753 | −0.020 |
| old `prm_verifiability_all8_majority` | 0.599 | 0.637 | 0.606 | 0.806 | +0.031 [−0.069, +0.125] | 0.733 | 0.753 | +0.226 |

The first pick gain of any classifier on this cache whose bootstrap CI excludes zero
(+0.076 ≈ 40% of the random → oracle gap; pick above the problem's mean on 11 of 15
problems; `rerank15_bootstrap.py`). Unlike every earlier classifier it prefers *longer*
trajectories (ρ with step count +0.23 vs −0.20 for the old verifiability one), so this is
not the §10.1 length artefact in reverse. But the picked trajectory is *less* often
correct than random (0.667 vs 0.753, −0.087 [−0.233, +0.053]), the same decoupling as in
8.3: on these problems selecting for formalizability does not select for correctness.

### 8.5 Bottom line

Encoder capacity was the binding constraint on the step judge: Qwen3-1.7B on the same
5,415 labels is the first classifier that beats the FaithformBench ones on *every* real-Lean
evaluation set, in and out of distribution (0.65–0.75 vs 0.58–0.69). It is **not** the
binding constraint on the decoding outcome: constrained beam search with it is null on
accuracy and shifts formalization by ~2–3 points (significant only on the granularity-
sensitive count), and full-trajectory rerank with it raises the picked trajectory's
formalization without raising its correctness. Two things remain to try before concluding
the mechanism is dead: (i) the α / `tau_phi` sweep of 8.3 point 3 — *done, §8.6: it costs
accuracy and adds no formalization*; (ii) a harder problem set
(MATH levels 4–5, where §5.4 found the Lean rate *does* predict correctness), since on
MATH-500 levels 1–3 the ceiling itself is near zero.

### 8.6 Intervention strength — α sweep (Oct 16): pushing harder costs accuracy, buys no formalization

§8.3 point 3 asked whether the Qwen3 judge's small effect was the calibration shrinking it
(T = 8.3). `beam_decode_v2.py` on the same 110 problems and settings, α ∈ {4, 10, 30}
(α = 4 matches the MiniLM arm's effective scale; 10 and 30 push harder); all arms
Lean-labelled (1,077 new unique steps, 3,624 total, 14.5 GPU-h). Paired against the
search-only control:

| arm | acc | L1 | L2 | L3 | acc vs `beam_lm` (W/L, p) | step rate | word share | proved steps | words/sol | final ≠ `beam_lm` |
|---|---|---|---|---|---|---|---|---|---|---|
| `beam_lm` | 0.655 | 0.727 | 0.756 | 0.512 | — | 0.526 | 0.522 | 4.26 | 108 | — |
| `beam_forked_qwen3` (α = 1) | 0.627 | 0.727 | 0.733 | 0.465 | 8/11, p = 0.65 | 0.557 | 0.543 | 4.76 | 84 | 88/110 |
| `beam_forked_qwen3_a4` | 0.564 | 0.727 | 0.622 | 0.419 | 8/18, **p = 0.076** (Wilcoxon 0.050) | 0.558 (p = 0.11) | 0.544 (p = 0.31) | 4.97 (p = 0.075) | 83 | 105/110 |
| `beam_forked_qwen3_a10` | 0.582 | 0.727 | 0.578 | 0.512 | 14/22, p = 0.24 | 0.546 (p = 0.40) | 0.531 (p = 1.00) | 4.80 (p = 0.12) | 82 | 106/110 |
| `beam_forked_qwen3_a30` | 0.555 | 0.591 | 0.556 | 0.535 | 13/24, **p = 0.099** | — | — | — | 80 | 108/110 |

(α = 30 decoded for accuracy only. Sign-test p for the formalization columns is against
`beam_lm`; a4 vs a10 differ on nothing, p ≥ 0.19.)

Reading: a dose–response in the wrong direction. Every increase in α changes more final
answers (80% → 95–98% of problems) and lowers accuracy — by 7–10 points at α ≥ 4, at the
edge of significance on 110 problems — while formalization stays where α = 1 left it
(step rate +2–3 points, word share +1–2, nothing significant). The judge's steering
saturates on formalization at α ≈ 1 and then only trades correctness away. So the
calibration was not shrinking a useful signal: the signal itself is orthogonal to (or
mildly against) correctness on these problems, consistent with §5.4 / §8.4. Intervention
strength is closed as a lever on MATH-500 levels 1–3.

## 9. What "not proved" means — refutation probe (Oct 16)

`experiments_v1.md` §4 left open whether the real-Lean "proved" metric measures
*formalizability* (the claim can be stated and proved) or *truth* (a false claim fails
regardless of shape). With statements now stored for every label, `negation_probe.py`
asked DeepSeek-Prover-V2 (same settings: 8 samples, 60 s, maxHeartbeats 400000) to prove
the **negation** of the stored statement under its own binders and hypotheses — a
refutation — for 100 prove-stage failures from each label set (seed 0) and 30 proved
statements per set as a control. 4.7 GPU-h; `logs/negation_probe.json`.

| label set (generator / problems) | failures probed | refuted | controls (proved statements) refuted |
|---|---|---|---|
| forked trees (step-native / MATH-train 1–5) | 100 | 4 | 0 / 30 |
| exp-2 beam arms (step-native / MATH-500 1–3) | 100 | 3 | 0 / 30 |
| rerank cache (natural CoT / MATH-500 1–3) | 100 | 0 | 0 / 30 |
| **all** | **300** | **7 (2.3%)** | **0 / 90** |

All 300 statements were negatable (parsed into binders + conclusion). The seven refutations
are concrete computations gone wrong — `f 2 = 86` for `x⁴+5x³+9x²−x−14`, a polar-to-
Cartesian conversion, two cross products, an algebraic-system step — closed by `norm_num`
/ `simp` / `linarith`. 196 of the 300 failures are closed numeric claims (no binders),
exactly the kind the prover refutes most easily when they are false, and only 4 of those
196 were refuted.

Reading: **"not proved" is search / statement failure, not demonstrated falsity.** At most
a few percent of failing steps can be shown false by the same prover; the rest are
either true and too hard for an 8-sample tactic search, or formalized into something the
prover can neither prove nor refute (under-specified `let`-chains, real-analysis facts,
geometry). The refuted share is a lower bound on falsity — the prover's refutation search
fails for the same reasons its proof search does — but the asymmetry on closed numeric
claims (easy to refute when false, 2% refuted) says most failing steps are not simply
wrong arithmetic. So the metric's name should be read as *provable by this pipeline*:
it does carry some truth signal (correct trajectories have higher step rates, §4.1 /
§5.4), but a classifier that learns it is learning "which claims this prover closes",
which is why selecting on it shortens and concretises steps (§8.3, §8.6) rather than
fixing answers.

## 10. Lean in the loop — hybrid beam search gated by the Qwen3 judge (Oct 16–17)

The one decoding mechanism that does not depend on the judge's AUC: `lean_constrained_beam_search`
with `INTERVENE_HYBRID`. Each candidate step whose calibrated probability under
`forked_supervised_qwen3` is ≥ 0.25 gets the classifier term (α = 1, mean aggregation, as
in §8.3); below 0.25 it is sent to real Lean (same pipeline and settings as every label in
this file), and a failed check scores −50 (the beam is pruned). A per-problem budget of
24 real Lean calls; a first unbudgeted run was spending up to 61 calls on single problems
and was stopped at 10 problems. `hybrid_beam_decode.py`; 110 easy problems, 18.1 GPU-h
decode + 13.1 GPU-h to label the final trajectories (verdicts obtained *during* decoding
are reused as labels — same pipeline, same settings). Arm `beam_hybrid_qwen3_t0.25_b24`.

Lean usage: 1,260 real calls (11.5 per problem; 30.9% of checked candidates proved); the
judge asked for 6,391 escalations, so the budget bound on 33 problems; 30 problems never
escalated (on those the decode is identical to `beam_forked_qwen3`, 30/30 same outcome).

| arm | acc | acc vs `beam_lm` (W/L, p) | steps/sol | words/step | step rate | word share | proved steps |
|---|---|---|---|---|---|---|---|
| `beam_lm` | 0.655 | — | 9.3 | 11.6 | 0.526 | 0.522 | 4.26 |
| `beam_forked_qwen3` (judge only) | 0.627 | 8/11, p = 0.65 | 9.3 | 9.0 | 0.557 | 0.543 | 4.76 |
| `beam_hybrid_qwen3_t0.25_b24` | 0.645 | 12/13, p = 1.00 | **15.0** | 7.9 | **0.601** | **0.601** | **5.43** |

Paired vs `beam_lm` (110 problems): step rate 50/31, Wilcoxon **p = 0.001**; word share
53/34, **p = 0.002**; proved steps 49/26, **p = 0.001**; correct 12/13, p = 0.84. Paired vs
the judge-only arm: word share 34/19, **p = 0.004**; step rate p = 0.007; correct 7/5. By
level the word-share gain is +13 (L1, p = 0.02), +7 (L2, p = 0.08), +6 (L3, p = 0.11)
points. **This is the first arm in the project whose real-formalization gain is
significant on the granularity-robust measure, at unchanged accuracy.**

Two caveats that qualify it:

1. **Per-problem vs pooled.** The table averages per problem. Pooled over all steps the
   picture inverts: hybrid 0.361 step rate / 0.391 word share vs `beam_lm` 0.458 / 0.370
   and judge-only 0.513 / 0.503. The hybrid arm's gain lives on short solutions; its long
   solutions carry many unproved steps.
2. **The budget fallback was degenerate.** Split by Lean usage:

   | group | n | acc `beam_lm` → hybrid | word share `beam_lm` → hybrid (Wilcoxon p) | steps/sol `beam_lm` → hybrid |
   |---|---|---|---|---|
   | no Lean call | 30 | 0.733 → 0.800 | 0.663 → 0.721 (0.11) | 5.9 → 6.2 |
   | Lean used, within budget | 47 | 0.723 → 0.723 | 0.526 → 0.606 (**0.064**) | 7.9 → 7.5 |
   | budget hit | 33 | 0.485 → **0.394** | 0.387 → 0.484 (0.048) | 14.4 → **33.8** |

Once the 24 calls were spent, unsure steps were *trusted* (score 0 instead of the
   classifier's negative logit), so the search stopped penalising doubtful steps and ran
   to 34-step solutions on the 33 hardest problems, losing 3 answers there. Fixed:
   `_apply_verifier` now falls back to the classifier term when the verifier declines a
   step (`tests/test_beam_search.py`), and the arm was re-run from scratch — below.

### 10.1 Corrected run — `beam_hybrid_qwen3_t0.25_b24f` (Oct 18–19)

Same threshold and budget; declined steps get the classifier term. 27.9 GPU-h decode
(1,257 real Lean calls, 11.4 per problem; the judge asked for 2,607, budget bound on the
same 33 problems) + 4.3 GPU-h to label the 318 final-trajectory steps not already
verified during decoding. All 110 problems fully labelled in every arm (6,426 labels).

| arm | acc | acc vs `beam_lm` (W/L) | steps/sol | words/sol | step rate | word share | proved steps | pooled step rate | pooled word share |
|---|---|---|---|---|---|---|---|---|---|
| `beam_lm` | 0.655 | — | 9.3 | 108 | 0.526 | 0.522 | 4.26 | 0.458 | 0.370 |
| `beam_forked_qwen3` (judge only) | 0.627 | 8/11 | 9.3 | 84 | 0.557 | 0.543 | 4.76 | 0.513 | 0.503 |
| **`beam_hybrid_qwen3_t0.25_b24f`** | **0.655** | 11/11 | 9.4 | 85 | **0.588** | **0.580** | **5.16** | **0.551** | **0.527** |

Paired vs `beam_lm`: step rate 48/32, Wilcoxon **p = 0.006**; word share 57/31, sign
**p = 0.007**, Wilcoxon **p = 0.004**; proved steps 51/23, **p = 0.002**; correct 11/11,
p = 1.0. Paired vs the judge-only arm: word share 30/20, **p = 0.038**; step rate p = 0.08;
correct 8/5. By level vs `beam_lm` the word-share gain is +13 points at level 1
(p = 0.002), +4 at level 2 (p = 0.14), +4 at level 3 (p = 0.27). The runaway is gone
(9.4 steps/solution; budget-hit problems 15.2 vs 14.8 for `beam_lm`, accuracy 0.455 =
0.455 there), and the per-problem and pooled measures now agree.

Split by Lean usage (word share `beam_lm` → hybrid): no Lean call (30) 0.663 → 0.721;
Lean within budget (47) 0.534 → 0.575; budget hit (33) 0.376 → 0.458 (p = 0.046).

**Reading (final).** Real Lean as a pruning oracle inside beam search is the one
intervention in this project that raises the verified content of the final solution
significantly — +6 points of proved step rate and +6 points of proved word share over
the search alone, +4 over the best learned judge, on the granularity-robust measure,
with answer accuracy exactly where the search alone puts it (0.655 = 0.655). The learned
judge's role is to decide *which* steps are worth a Lean call: ~11 calls per problem
(≈ 0.25 GPU-h/problem at 8 prover samples) instead of the ~70 candidates a beam of 4 × 2
generates. On MATH-500 levels 1–3 this buys formalization, not correctness — the same
decoupling as every other arm on these problems (§5.4, §8.4, §8.6); §11 is where the
coupling appears.


## 11. Hard problems — MATH-500 levels 4–5 (Oct 15–18)

§5.4 predicted that the Lean rate would carry answer signal only where problems are hard
enough for wrong trajectories to go wrong *visibly*. Two experiments on the level-4/5 half
of MATH-500 (131 problems, none used to fit any classifier): full-trajectory rerank on
forked trees (§11.1), and the beam arms (§11.2, labels still running).

### 11.1 Full-trajectory rerank on 60 hard forked trees — the first accuracy gain

`hard_forked_sample.py`: 30 level-4 + 30 level-5 MATH-500 problems (seed 0), 4 forked
trajectories each from the step-native Qwen2.5-7B (240 trajectories, 6,780 steps, 3,786
unique, 44% Lean saving; 219/240 reached an answer; 74/240 = 31% correct). Every unique
step real-Lean labelled (facts rendering, 8 samples, 60 s): **14.0% proved** (vs 27.6% on
the MATH-train trees and 46–58% on MATH-500 levels 1–3). 2.0 GPU-h sampling + 53.7 GPU-h
Lean (two 48-h-capped jobs plus the resume). Analysis `hard_forked_analyse.py`,
`hard_forked_bylevel.py`; `hard/report_hard_rerank.json`.

**The ceiling exists here.** Picking each problem's trajectory by its *true* Lean rate
gives accuracy 0.367 vs random 0.308 (any-correct oracle 0.467); within-problem
ρ(true rate, correct) = **+0.35** over the 22 problems with variation in both; correct
trajectories prove 0.458 of their steps vs 0.299 for incorrect ones. On the easy rerank
cache the same oracle pick was *below* random (§4.1). So the decoupling of §4.1/§8 is a
property of easy problems, as §5.4 said.

**And the Qwen3 judge reaches it.** Per problem, pick the trajectory with the highest
mean step probability (same protocol as §4.1 / §8.4; CIs: 5,000-sample bootstrap over
problems):

| policy | step AUC | sibling acc | PRM pick (true rate) | random | oracle | pick − random [95% CI] | within-problem ρ | **pick acc** | random acc | oracle acc | **acc − random [95% CI]** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `forked_supervised_qwen3` | **0.742** | 0.429 | **0.408** | 0.348 | 0.466 | **+0.060 [+0.032, +0.090]** | **+0.267** | **0.383** | 0.308 | 0.467 | **+0.075 [+0.013, +0.138]** |
| `forked_supervised` (MiniLM) | 0.695 | 0.671 | 0.369 | 0.348 | 0.466 | +0.021 [−0.014, +0.058] | +0.167 | 0.333 | 0.308 | 0.467 | +0.025 [−0.058, +0.108] |
| old `prm_verifiability_all8_majority` | 0.651 | 0.614 | 0.354 | 0.348 | 0.466 | +0.006 [−0.023, +0.034] | +0.159 | 0.283 | 0.308 | 0.467 | −0.025 [−0.096, +0.042] |

By level (Qwen3 pick): level 4 — accuracy 0.467 vs random 0.375 (oracle 0.500), true rate
0.438 vs 0.382; level 5 — accuracy 0.300 vs 0.242 (oracle 0.433), true rate 0.378 vs 0.313.
The pick beats the problem's random correctness on 17 problems, ties on 38 (all four
trajectories agree), trails on 5. It is not a length artefact: the picked trajectory is
*shorter* than the problem's mean at level 4 (19 vs 26 steps) and *longer* at level 5
(36 vs 31).

Reading:

1. **This is the first selection gain in correctness in the project**, and it comes from
   exactly the combination the preceding sections isolated: a judge trained on the
   generator's own Lean-labelled steps (§7), with enough capacity to transfer (§8), applied
   where formalizability and correctness are coupled (§5.4, here). The Qwen3 pick recovers
   half the random → true-rate-oracle gap in formalization (0.348 → 0.408 of 0.466) and
   almost half the random → any-correct gap in accuracy (0.308 → 0.383 of 0.467); the pick
   is even slightly *above* the true-rate oracle's accuracy (0.367), i.e. the judge's
   score is as good a correctness proxy as the ground-truth Lean rate on this data.
2. **Neither weaker judge gets there.** MiniLM on the same data: +0.025 accuracy, CI
   through zero; the FaithformBench classifier: −0.025. Step AUC orders them (0.742 /
   0.695 / 0.651) and so do the pick gains — on hard problems the AUC *does* predict the
   selection gain, unlike on easy ones (§4.1), because here there is a selection signal
   to be had.
3. **Size.** +0.075 is 4.5 problems of 60; the CI's lower end is +0.013 (one problem).
   One split, one base model, one seed. It needs the 110-problem beam set (§11.2) and,
   ideally, a second sample of hard problems before it is a headline number rather than
   a strong lead.

### 11.2 Beam arms on 110 hard problems (Oct 15–19) — formalization yes, accuracy no

`beam_decode_v2.py --set hard`: the first 110 of the 131 level-4/5 MATH-500 problems
(64 level 4, 46 level 5), `greedy` / `beam_lm` / `beam_forked_qwen3` with the §5.1 settings
(beam 4 × 2, α = 1, mean aggregation; `tau_phi` from the held-out MATH-train trees as
before). 4 h decode; **5,097 unique steps** in the two beam arms (hard solutions are
2.5× longer), real-Lean labelled over three 48-h-capped jobs, 92 GPU-h in all;
**20.5% proved**. Analysis as §5.2 / §5.3.

| arm | acc | L4 | L5 | steps/sol | words/step | words/sol | final ≠ `beam_lm` | vs `beam_lm` (W/L, p) | vs `greedy` (W/L, p) |
|---|---|---|---|---|---|---|---|---|---|
| `greedy` | 0.255 | 0.328 | 0.152 | 16.6 | 12.0 | 199 | 105/110 | 10/24, **p = 0.02** | — |
| `beam_lm` | **0.382** | 0.438 | 0.304 | 23.3 | 10.7 | 249 | — | — | 24/10, **p = 0.02** |
| `beam_forked_qwen3` | 0.373 | 0.406 | 0.326 | 25.4 | 7.9 | 200 | 84/110 | 7/8, p = 1.0 | 22/9, **p = 0.03** |

| formalization, paired per problem | `beam_lm` | `beam_forked_qwen3` | better / worse | sign p | Wilcoxon p |
|---|---|---|---|---|---|
| step rate (proved / steps) | 0.371 | 0.404 | 47 / 38 | 0.39 | **0.040** |
| word share (words in proved steps) | 0.342 | 0.386 | 51 / 39 | 0.25 | **0.018** |
| proved steps per solution | 4.56 | 6.01 | 48 / 31 | 0.07 | **0.003** |
| correct | 0.382 | 0.373 | 7 / 8 | 1.0 | 0.80 |

By level: word share +5 at level 4 (p = 0.03), +4 at level 5 (p = 0.18); accuracy 3/5 and
4/3. Step AUC of the judges on these 5,097 steps: `forked_supervised_qwen3` **0.709** (0.730 on the
`beam_lm` steps it did not pick, 0.696 on its own), old FaithformBench verifiability 0.553
(0.511 / 0.599) — the Qwen3 judge transfers to hard, long, 20%-proved solutions as well as
it did to the easy set (0.720, §8.2); the benchmark-trained one is back near chance.

Reading:

1. **Beam search is worth +13 accuracy points on hard problems** (greedy 0.255 →
   `beam_lm` 0.382, p = 0.02) — twice its easy-set gain. The classifier arm keeps that
   gain (0.373, p = 0.03 vs greedy) and adds nothing to it (7/8 vs `beam_lm`).
2. **The judge's formalization gain, null on easy problems (§8.3), is significant on
   hard ones** — word share +4.4 points (p = 0.018), proved steps +1.4 (p = 0.003) — at
   unchanged accuracy and with 20% shorter solutions (200 vs 249 words). On hard problems
   there is more unproved material for the judge to steer away from.
3. **Trajectory-level selection and step-level steering part ways on hard problems.**
   The same judge that picks the correct complete trajectory 7.5 points more often than
   chance (§11.1) does not move accuracy when it steers the beam one step at a time. The
   difference is the decision: choosing among 4 finished solutions with their final
   answers in view (where the judge's mean score correlates +0.27 with the true Lean rate
   and that correlates +0.35 with correctness) vs choosing among 8 partial continuations
   that share most of their prefix. The mean-step-probability of a *complete* solution
   is a usable correctness proxy on hard problems; a per-step logit inside the search is
   not, at least at α = 1 with this calibration.

So the hard-problem picture is: formalizability is coupled to correctness (§11.1
ceiling), the judge sees it (AUC, pick gain), and the way to cash it in is to rerank
complete solutions — which is also the cheap operation (score N finished solutions once)
rather than the expensive one (score 8 candidates per step).

## 12. Internal-state probes as the step judge (Oct 25)

`~/LeanSteer` fits linear probes on the residual stream of the four FaithformBench
autoformalizers that predict the benchmark's prover verdict for a step *before the model
writes any Lean* (Goedel: proved 0.75, proved_strict 0.80, sycophancy 0.78 AUROC on the
benchmark). Our real-Lean pipeline uses the same formalizer, Goedel-Formalizer-V2-8B, so a
probe on its hidden state is a judge that reads the formalizer itself rather than a
separate text classifier. Question: does it transfer to our generator's steps, and can it
steer the constrained beam search?

**Data.** Every real-Lean-labelled unique step of the project, 20,964 in all
(`probe_build_steps.py`): forked trees 5,415 (standard 120/30 split), hard forked trees
3,786, easy beam arms 5,149, hard beam arms 5,058, natural-CoT rerank cache 1,556. Goedel's
residual stream at layers 0, 2, …, 36 and two anchors (`prompt_last`, `step_mean`) under two
renderings of the step: **bare** (the benchmark's prompt around the step text — what the
af_v1 probes were fit on) and **facts** (`render_step_claim` with the previous steps as
hypotheses — the exact claim our pipeline formalized to produce the label). 0.8 GPU-h
(`probe_collect.py`, LeanSteer's `HFBackend`); `probe/acts/`.

### 12.1 Offline: AUC against our labels (`probe_eval.py`, `probe_eval2.py`)

| judge | fit on | forked held | exp2 (easy beam) | hard beam | hard forked | natural CoT |
|---|---|---|---|---|---|---|
| af_v1 Goedel probe, proved, step_mean L36, **bare** | FaithformBench | 0.638 | 0.644 | 0.580 | 0.614 | 0.582 |
| af_v1 Goedel probe, proved_strict, prompt_last L36, bare | FaithformBench | 0.633 | 0.529 | 0.522 | 0.626 | 0.557 |
| af_v1 probes, any task/anchor, **facts** rendering | FaithformBench | 0.47–0.67 | 0.45–0.57 | 0.32–0.59 | 0.23–0.69 | 0.47–0.58 |
| in-generator probe, bare, prompt_last L20 | forked fit (4,435) | 0.702 | 0.692 | 0.624 | 0.692 | 0.509 |
| in-generator probe, facts, prompt_last L34 | forked fit | 0.677 | 0.667 | 0.692 | 0.737 | 0.606 |
| **in-generator probe, facts, prompt_last + step_mean, L12+20+28+36** (val-selected) | forked fit | 0.699 | 0.707 | 0.692 | **0.778** | 0.583 |
| in-generator probe, facts, prompt_last, L8+14+20+26+34 | forked fit | 0.704 | 0.683 | 0.704 | **0.831** | 0.670 |
| external MiniLM `forked_supervised` (§7.4, §8.2) | forked fit | 0.692 | 0.631 | — | — | 0.572 |
| external **Qwen3-1.7B** `forked_supervised_qwen3` (§8.2, §11) | forked fit | **0.753** | **0.720** | **0.709** | 0.742 | **0.654** |

Reading:

1. **The FaithformBench-trained probes do not transfer**, exactly like the FaithformBench
   external classifiers (§10.2): 0.58–0.64 on the bare rendering, at or below chance on the
   facts rendering (a prompt distribution they never saw). The "this step is provable"
   direction LeanSteer found is specific to the benchmark's steps and prompt.
2. **Refit on the generator's own Lean-labelled steps, the probe transfers** — 0.67–0.78
   out of distribution — but it does not beat the external Qwen3-1.7B classifier trained on
   the same 4,435 steps except on the hard forked trees (0.778 vs 0.742; 0.831 for a
   variant that scored lower on validation). On easy MATH-500 steps it trails by 0.01–0.02,
   on natural chain-of-thought by 0.05–0.07. Capacity of the read-out (a 32k-dimensional
   linear probe on 3.7k rows, C = 0.05) and the fact that the probe reads *Goedel's*
   state rather than a model fine-tuned on the task are the obvious reasons; a larger
   training pool (forked + easy beam + natural CoT, ~11k rows) did not raise the hard-set
   AUCs (0.70–0.71 / 0.80–0.81).
3. **Rendering matters in the direction the labels predict.** Facts beats bare on every
   out-of-distribution set except the easy beam steps; bare collapses on natural CoT
   (0.51), whose steps lean on context the bare prompt removes. `prompt_last` (the state
   from which Goedel starts writing) beats `step_mean` out of distribution.

Full-trajectory rerank on the 60 hard forked trees with the selected probe (mean
probability over the trajectory's steps; same protocol as §11.1): pick true rate 0.378 vs
random 0.348, **+0.030 [+0.004, +0.057]**; pick accuracy 0.367 vs random 0.308,
**+0.058 [−0.004, +0.121]** — the same direction as the Qwen3 classifier (+0.060 / +0.075),
a shade smaller, the accuracy CI just touching zero.

### 12.2 Probe-guided constrained beam search — running

`rllf/policy/probe_policy.py::GoedelProbePolicy` wraps Goedel through LeanSteer's hooked
backend and exposes `predict_step_probability` / `predict_step_logit_for_beam` so
`lean_constrained_beam_search` takes it in place of an external classifier (one Goedel
prefill per candidate step, ~50 per problem; `tests/test_probe_policy.py` checks the
feature order against the offline fit). Arm `beam_probe_goedel`: the validation-selected
probe above, α = 1, mean aggregation, `tau_phi` fitted on the 980 held-out tree steps
(T = 7.30, b = 7.0 — the probe's logits are on a wider scale than the classifier's), on the
easy 110 and the hard 110 problems, then real-Lean labels (`probe_beam_decode.py`,
`launch_probe_beam.sh`). Decode 25 s / ~50 probe calls per easy problem; both sets decoded in
one job (48 GPU-h including the easy-set labels and most of the hard-set labels; hard-set
labelling resumed in a second job).

**Easy 110 (levels 1–3), all labelled:**

| arm | acc | acc vs `beam_lm` (W/L, p) | steps/sol | words/sol | step rate | word share | proved steps |
|---|---|---|---|---|---|---|---|
| `beam_lm` | 0.655 | — | 9.3 | 108 | 0.526 | 0.522 | 4.26 |
| `beam_forked_qwen3` (external judge) | 0.627 | 8/11, p = 0.65 | 9.3 | 84 | 0.557 | 0.543 | 4.76 |
| `beam_probe_goedel` (internal probe) | 0.600 | 9/15, p = 0.31 | 10.4 | 92 | 0.536 (p = 0.77) | 0.514 (p = 0.68) | 4.86 (p = 0.10) |

Paired vs the external Qwen3 arm: word share 39/51, Wilcoxon p = 0.076 (lower); step rate
p = 0.28; correct 9/12. The probe changes the final answer on 65/110 problems — it steers —
but toward steps that prove no more often than the search's own, at a 5-point accuracy
cost (n.s.). On easy problems the internal read-out is a weaker judge than the external
classifier, consistent with its lower offline AUC there (0.707 vs 0.720).

**Hard 110 (levels 4–5):** decoded (accuracy 0.364 vs `beam_lm` 0.382, 6/8, p = 0.79; final
answer changed on 83/110; 20.2 steps / 193 words per solution vs 23.3 / 249); real-Lean
labels complete (6,928 labels in the hard arms, 20.0% proved; 5.6 GPU-h for the resumed pass):

| hard 110, paired per problem | `beam_lm` | `beam_forked_qwen3` | `beam_probe_goedel` | probe vs `beam_lm` (W/L, sign p, Wilcoxon p) | probe vs Qwen3 arm |
|---|---|---|---|---|---|
| step rate | 0.371 | 0.404 | 0.403 | 50/37, 0.20, 0.12 | 39/44, p = 0.61 |
| word share | 0.342 | 0.386 | 0.379 | 55/38, 0.10, **0.083** | 37/55, p = 0.29 |
| proved steps / solution | 4.56 | 6.01 | 4.78 | 42/36, 0.57, 0.26 | 27/45, **p = 0.004** (shorter solutions) |
| correct | 0.382 | 0.373 | 0.364 | 6/8, p = 0.79 | 6/7, p = 1.0 |

By level the probe's word-share gain over `beam_lm` is +3.6 (level 4) and +3.9 points
(level 5), neither significant; accuracy 1/5 at level 4, 5/3 at level 5.

### 12.3 Reading

1. **The internal probe reproduces the external classifier's effect on hard problems and
   falls short of it on easy ones.** Hard: the same +3–4 points of formalization (word
   share 0.379 vs the Qwen3 arm's 0.386, indistinguishable) at unchanged accuracy, with
   22% shorter solutions. Easy: no formalization gain at all (word share 0.514 vs 0.543 for
   the Qwen3 arm, p = 0.08) and a 5-point accuracy dip (n.s.). This tracks the offline AUCs
   exactly (hard 0.69–0.78 vs 0.71–0.74; easy 0.707 vs 0.720).
2. **As a judge, the formalizer's own state is usable but not privileged.** The hypothesis
   was that Goedel's residual stream, read from the exact prompt it formalizes, would know
   better than any external model what it is about to prove. On these data it knows about
   as much as a 1.7B text classifier fine-tuned on the same 4,435 labels, and less than it
   on steps far from the training distribution (natural chain-of-thought 0.58 vs 0.65). The
   pre-generation state is "informative but far from decisive" in LeanSteer's own words;
   the post-output state (0.86 there) would need a generation per candidate, i.e. most of
   the cost of just running the Lean pipeline.
3. **Nothing changes in the programme's conclusions.** Step-level steering — by an
   external classifier, by an internal probe, at any α — does not move accuracy on either
   difficulty band; real Lean in the loop raises verified content; trajectory-level
   selection on hard problems is where a judge turns into answers, and the probe's rerank
   gain there (+0.058 [−0.004, +0.121]) is the Qwen3 classifier's (+0.075 [+0.013, +0.138])
   with a slightly wider margin of doubt.
4. **What would make the probe worth more.** (a) Post-output anchors: one Goedel
   generation per candidate (~200 tokens of thinking plus the statement) makes the verdict
   far more readable (0.82–0.86 on the benchmark) — about 4× the prefill cost, still 20×
   cheaper than a prover call; (b) a probe trained on more and harder in-generator labels
   (the 20,964-step pool here is problem-disjoint from the hard sets only if fit on the
   forked trees alone); (c) steering Goedel itself along the probe direction at
   formalization time, which LeanSteer showed is causal for Kimina but not for Goedel.

### 12.4 Attention-head probes (Oct 28)

LeanSteer's second probe family reads the per-head attention outputs (the input of each
block's `o_proj`, 32 heads × 128 for Goedel) instead of the residual stream, and on
FaithformBench concatenating the 4–64 best heads beats the residual probe by 0.05–0.11
(Goedel proved 0.80 vs 0.74, proved_strict 0.86 vs 0.77, sycophancy 0.84 vs 0.78). Same
collection as §12 but `component="heads"`, all 36 layers (`probe_collect_heads.py`, 0.8 GPU-h,
24 GB); `GoedelProbePolicy` extended to head slices (`component=heads`, tested).

**A. FaithformBench-selected and -trained head probes, zero-shot** (LeanSteer saved the head
ranking but not the coefficients, so the top-k read-outs were refit on the benchmark's own head
store, prompt_last, bare rendering; `probe_eval_heads.py`):

| task, k heads | benchmark test | forked held | exp2 | hard beam | hard forked | natural CoT |
|---|---|---|---|---|---|---|
| proved, top-4 (L36H8, L36H9, L23H15, L36H12) | 0.802 | **0.697** | **0.660** | 0.621 | **0.759** | 0.591 |
| proved, top-16 | 0.805 | 0.664 | 0.624 | 0.598 | 0.705 | 0.579 |
| proved_strict, top-16 | 0.860 | 0.709 | 0.613 | 0.561 | 0.646 | 0.540 |
| sycophancy, top-4 | 0.820 | 0.659 | 0.588 | 0.598 | 0.730 | 0.577 |
| (residual af_v1 probes, §12.1) | 0.74–0.80 | 0.63–0.64 | 0.53–0.64 | 0.52–0.58 | 0.61–0.63 | 0.56–0.58 |

Four benchmark-selected heads transfer to our steps **better than any benchmark-trained
residual probe** (0.70 / 0.66 / 0.76 vs 0.64 / 0.64 / 0.61 on held / easy / hard trees) — the
only FaithformBench-trained judge in the project that is clearly above chance out of
distribution. More heads transfer worse: the benchmark-specific part grows with k.

**B. In-generator head probes** (`probe_eval_heads.py` B): selecting heads on our own 699-row
validation fifth overfits — 1,129 of 1,152 heads clear 0.6 there, validation AUC of the
top-64 reaches 0.82, and the chosen probe scores 0.59 on the hard sets. **Benchmark-selected
heads, read-out refit on our steps** (`probe_eval_heads2.py`) removes the selection problem:

| heads (prompt_last ranking) | rendering / anchor | val | forked held | exp2 | hard beam | hard forked | natural CoT |
|---|---|---|---|---|---|---|---|
| **proved_strict top-64** (selected by validation) | facts / prompt_last | 0.761 | 0.707 | 0.697 | 0.686 | 0.771 | 0.586 |
| proved top-16 | facts / prompt_last | 0.734 | 0.712 | 0.716 | 0.692 | 0.806 | 0.609 |
| sycophancy top-64 | facts / prompt_last | 0.748 | 0.696 | 0.708 | 0.740 | 0.815 | 0.533 |
| proved_strict top-4 | bare / prompt_last | 0.715 | 0.667 | 0.654 | 0.677 | **0.831** | 0.542 |
| residual multi-layer probe (§12.1) | facts | 0.764 | 0.699 | 0.707 | 0.692 | 0.778 | 0.583 |
| external Qwen3-1.7B classifier | — | — | **0.753** | **0.720** | 0.709 | 0.742 | **0.654** |

Reading: on our data the head probes are **as good as the residual probe, not better** — the
0.05–0.11 head advantage LeanSteer measured on FaithformBench does not reappear (0.70 / 0.70 /
0.69 / 0.77 vs 0.70 / 0.71 / 0.69 / 0.78). Several head sets do beat every other judge on the
hard sets (sycophancy top-64 0.740 / 0.815; proved_strict top-4 0.831 on hard trees), but the
validation set (MATH-train trees) cannot see that, and picking them post hoc would be
selection on the test set. Hard-forked rerank with the validation-selected head probe: pick
true rate +0.010 [−0.024, +0.043], accuracy +0.042 [−0.042, +0.125] — weaker than the
residual probe and the Qwen3 classifier.

**C. Head-probe-guided beam search** — arm `beam_heads_goedel` with the validation-selected
head probe, same protocol as §12.2 (`tau_phi` T = 6.40, b = 5.84 on the held-out tree steps;
~50 probe calls per problem; 43 GPU-h for decode + labels of both sets).

| set / arm | acc | acc vs `beam_lm` (W/L, p) | steps/sol | words/sol | step rate | word share | proved steps |
|---|---|---|---|---|---|---|---|
| easy `beam_lm` | 0.655 | — | 9.3 | 108 | 0.526 | 0.522 | 4.26 |
| easy `beam_forked_qwen3` | 0.627 | 8/11 | 9.3 | 84 | 0.557 | 0.543 | 4.76 |
| easy `beam_probe_goedel` (residual) | 0.600 | 9/15 | 10.4 | 92 | 0.536 | 0.514 | 4.86 |
| easy **`beam_heads_goedel`** | 0.627 | 7/10, p = 0.63 | 12.0 | — | 0.539 (p = 0.54) | 0.518 (p = 0.91) | 5.37 (p = 0.58) |
| hard `beam_lm` | 0.382 | — | 23.3 | 249 | 0.371 | 0.342 | 4.56 |
| hard `beam_forked_qwen3` | 0.373 | 7/8 | 25.4 | 200 | 0.404 | 0.386 | 6.01 |
| hard `beam_probe_goedel` (residual) | 0.364 | 6/8 | 20.2 | 193 | 0.403 | 0.379 | 4.78 |
| hard **`beam_heads_goedel`** | 0.355 | 9/12, p = 0.66 | 21.3 | 195 | 0.389 (p = 0.27) | 0.368 (p = 0.17) | 4.38 (p = 0.41) |

Paired against the residual-probe arm the head arm differs on nothing (easy: word share 44/42,
p = 0.96; hard: 44/54, p = 0.55; correct 13/10 and 12/13). Against the Qwen3 arm on hard
problems: equal step rate and word share (46/46, 50/49), fewer proved steps because of
shorter solutions (p = 0.002), correct 10/12.

### 12.5 Reading — internal probes as beam judges

Three judges built from Goedel's own state have now steered the search — a FaithformBench
residual probe (not run: at chance offline), an in-generator residual probe (§12.2) and an
in-generator head probe with benchmark-selected heads (§12.4) — next to the external
MiniLM and Qwen3 classifiers. The outcome table is flat:

| judge, beam α = 1 | offline AUC (easy / hard beam steps) | easy acc | easy word share | hard acc | hard word share |
|---|---|---|---|---|---|
| none (`beam_lm`) | — | 0.655 | 0.522 | 0.382 | 0.342 |
| external Qwen3-1.7B | 0.720 / 0.709 | 0.627 | 0.543 | 0.373 | **0.386** (p = 0.018) |
| residual probe, 4 layers × 2 anchors | 0.707 / 0.692 | 0.600 | 0.514 | 0.364 | 0.379 (p = 0.083) |
| head probe, 64 benchmark heads | 0.697 / 0.686 | 0.627 | 0.518 | 0.355 | 0.368 (p = 0.17) |

1. **No judge moves accuracy; every judge moves hard-problem formalization by roughly its
   AUC margin over chance.** The three formalization gains on hard problems (+4.4, +3.7, +2.6
   points of word share) order exactly as the judges' hard-step AUCs (0.709, 0.692, 0.686).
   What a beam judge buys here is a property of how well it ranks steps, not of where it
   reads them from.
2. **Reading the formalizer's internals buys no information the step text does not carry.**
   On FaithformBench the heads beat the residual stream by 0.05–0.11 and the residual stream
   beat text baselines by 0.1–0.2; on this generator's steps, with labels from the exact
   prompt the probes read, head ≈ residual ≈ a fine-tuned 1.7B text classifier (±0.02), and
   all three trail the classifier on natural chain-of-thought. The one genuine advantage of
   the internal read-out is **zero-shot transfer of the head *selection***: four
   benchmark-chosen heads carry 0.70 / 0.76 to our easy / hard steps where every
   benchmark-trained classifier and residual probe carried ≈ 0.6.
3. **Cost.** A probe call is one Goedel prefill (8B, ~400 tokens): ~5× the Qwen3 classifier's
   forward pass, ~1/20 of a real Lean call. It is the cheapest way to get a 0.70-AUC judge
   *without training anything but a logistic regression* — the practical case for probes is
   label efficiency, not accuracy.

Programme conclusion unchanged: step-level steering by any judge leaves accuracy where the
search puts it; Lean in the loop raises verified content; trajectory-level reranking on hard
problems is where a judge becomes answers, and there the external Qwen3 classifier
(+0.075 [+0.013, +0.138]) remains ahead of the residual (+0.058) and head (+0.042) probes.

### 12.6 The generator's own state as a Lean judge (Nov 1)

Same recipe on the *step-native generator's* residual stream (Qwen2.5-7B SFT, teacher-forced on
the step: anchors `step_last` / `step_mean`, layers 0–28; `probe_collect_generator.py`, 0.6 GPU-h;
`probe_eval_gen.py`). Held-out trees 0.66–0.72 at the best layers, but out of distribution
0.55–0.67 (easy beam), 0.54–0.67 (hard beam), 0.57–0.73 (hard trees), 0.45–0.61 (natural CoT);
the validation-selected variant (`step_mean` L8) is among the worst OOD (0.55–0.59). The
formalizer's state (§12.1: 0.69–0.78) and the external classifier (0.65–0.75) both read Lean
verifiability better than the model that wrote the step does. Closed: the generator is not
where the Lean signal lives.

## 13. A weak verifier: Python-synthesised checks (Nov 2026, in progress)

**Motivation.** Lean formalization pays off only where provability tracks correctness (§11),
and most steps of a MATH solution are arithmetic or symbolic manipulations that a few lines of
Python / sympy can check directly — including *refuting* them, which the Lean pipeline almost
never does (§9: 2.3% of failures refutable). Plan: (1) a code model writes K = 4 independent
check programs per labelled step; sandboxed execution; aggregated verdict verified / refuted /
uncheckable / unknown; agreement with Lean and a number-perturbed sycophancy control; (2)
external classifiers and internal probes for *Python* verifiability; (3) a router python / lean /
none and a routed-verifier hybrid beam search; (4) SFT of the base model on verifiability-selected
trajectories in a JSON `[{step, verifier}]` format (MATH-train only). Scripts
`pyverify_generate.py` (vLLM; Qwen3-Coder-Next on two cards, Qwen3-30B-A3B-Instruct-2507 on one as
the fallback), `pyverify_execute.py` (import blacklist, `python -I`, 10 s CPU / 4 GB, verdict
line `VERDICT: TRUE|FALSE|UNCHECKABLE`); outputs under `pyverify*/`.

### 13.1 All 20,964 steps + 300 perturbed controls (Qwen3-30B-A3B-Instruct; Coder-Next run in progress)

85,056 programs (K = 4 per step; 1.2 + 1.7 GPU-h of generation, 20 CPU-min of sandboxed execution).
Program outcomes: TRUE 35 %, FALSE 26 %, no verdict line 15 %, runtime error 16 %, no code block 6 %,
timeout 0.7 %, UNCHECKABLE 0.3 %. The model essentially never declares a step uncheckable; a third
of steps end "unknown" through program failure, so checker quality (prompting, repair, more samples)
is the first lever. Aggregated verdicts (majority of ≥ 2 agreeing samples):

| set | n | verified | refuted | unknown | Lean proved | P(verified \| Lean) | P(verified \| ¬Lean) | P(Lean \| verified) | P(Lean \| refuted) |
|---|---|---|---|---|---|---|---|---|---|
| MATH-train trees | 5,415 | 0.387 | 0.270 | 0.341 | 0.276 | 0.614 | 0.300 | 0.438 | 0.170 |
| hard trees (L4–5) | 3,786 | 0.249 | 0.355 | 0.395 | 0.140 | 0.647 | 0.184 | 0.363 | 0.039 |
| easy beam arms (L1–3) | 5,149 | 0.535 | 0.226 | 0.237 | 0.414 | 0.698 | 0.420 | 0.540 | 0.135 |
| hard beam arms (L4–5) | 5,058 | 0.282 | 0.380 | 0.337 | 0.204 | 0.568 | 0.209 | 0.410 | 0.092 |
| natural CoT cache (L1–3) | 1,556 | 0.641 | 0.053 | 0.305 | 0.576 | 0.722 | 0.532 | 0.648 | 0.434 |
| **perturbed controls** (one number changed) | 300 | **0.120** | **0.543** | 0.333 | — | | | | |

**Sycophancy control.** Of the 129 controls whose *unaltered* source step was verified, the
altered version is verified 22.5 % of the time (29), refuted 44 % (57), unknown 33 %. So the checker
rubber-stamps roughly one false claim in five that it would have verified when true — a real
false-positive rate, but it catches wrong arithmetic in a way Lean's pipeline never does (§9: 2.3 %
of failures refutable).

**The weak verifier separates correct from wrong reasoning far better than Lean does**, on every
set. Steps that occur only in correct vs only in wrong solutions:

| set | only in correct: verified / refuted / Lean proved | only in wrong: verified / refuted / Lean proved |
|---|---|---|
| MATH-train trees | 0.652 / **0.109** / 0.429 (n = 945) | 0.319 / **0.316** / 0.232 (n = 4,114) |
| hard trees | 0.695 / **0.065** / 0.390 (n = 154) | 0.145 / **0.450** / 0.093 (n = 1,562) |
| easy beam arms | 0.666 / **0.070** / — (n = 1,869) | 0.457 / **0.329** / — (n = 3,072) |
| hard beam arms | 0.623 / **0.083** / — (n = 727) | 0.222 / **0.432** / — (n = 4,308) |

As a trajectory-selection oracle (pick the trajectory with the best score; §5.4 protocol):

| trees | score | pick acc | random | oracle | gain [95% CI] | within-problem ρ |
|---|---|---|---|---|---|---|
| MATH-train (150) | Lean proved rate | 0.373 | 0.318 | 0.547 | +0.055 [+0.005, +0.107] | +0.25 |
| MATH-train | Python verified rate | 0.380 | 0.318 | 0.547 | +0.062 [+0.013, +0.110] | +0.29 |
| MATH-train | Python verified − refuted | 0.407 | 0.318 | 0.547 | **+0.088 [+0.035, +0.140]** | +0.39 |
| MATH-train | fewest Python refutations | **0.413** | 0.318 | 0.547 | **+0.095 [+0.040, +0.148]** | **+0.44** |
| hard (60) | Lean proved rate | 0.367 | 0.308 | 0.467 | +0.058 [−0.013, +0.125] | +0.35 |
| hard (60) | any Python score | 0.367 | 0.308 | 0.467 | +0.058 [−0.004, +0.125] | +0.22 to +0.44 |

The extra signal is refutation — a Python check can say a step is *wrong*, which is what trajectory
selection needs and what "not proved" does not provide. On the 60 hard trees the four scores tie at
the same +0.058 (with only four trajectories per problem the picks coincide); the within-problem ρ
of "fewest refutations" (+0.44) is the highest of any score measured in this project.

### 13.2 Python verifiability is more readable from internal state than Lean verifiability

Probes on the stored activations (forked fit split, standard recipe; `probe_eval_python.py`),
target = the Python verdict, with the Lean target alongside on the same rows:

| features | target | val | forked held | hard trees (partial) |
|---|---|---|---|---|
| generator state (step_last + step_mean, L12+18+24) | **python verified** | **0.802** | **0.781** | 0.796 |
| generator state | python refuted | 0.732 | 0.685 | 0.599 |
| generator state | lean proved | 0.700 | 0.701 | 0.665 |
| Goedel residual (facts, 4 layers × 2 anchors) | python verified | 0.738 | 0.717 | 0.805 |
| Goedel residual | lean proved | 0.764 | 0.703 | 0.830 |
| Goedel heads (benchmark proved_strict top-64) | python verified | 0.759 | 0.692 | **0.876** |
| Goedel heads | python refuted | 0.694 | 0.652 | 0.816 |
| cross-target: Lean-trained probe → Python labels (generator / Goedel residual / heads) | | | 0.68 / 0.67 / 0.67 | 0.64 / 0.77 / 0.82 |
| cross-target: Python-trained probe → Lean labels | | | 0.69 / 0.65 / 0.66 | 0.75 / 0.75 / 0.86 |

**The generator knows whether its step will check out in Python** (0.78 held-out, 0.80 on hard
problems) far better than whether it will prove in Lean (0.70 / 0.67) — the first probe target on
which the model that wrote the step beats the formalizer's state. Refutation is less readable
(0.69 / 0.60). The Python and Lean targets share most of a direction (cross-target 0.65–0.86), as
expected for two readings of "is this step right". (The hard-tree column is on the 1,753 steps
done so far and is inflated by that set's many wrong steps; re-read at full data.)

### 13.3 Routing between verifiers (first version, forked trees)

Heuristic label per step — python if the Python check verified it, lean if not but Lean proved it,
none otherwise — is 38 / 12 / 50 % on the fit split, 40 / 7 / 53 % held out and 20 / 5 / 74 % on
hard trees. A flat three-way router (`router_eval.py`) learns mostly to say "none": the
generator-state router reaches 0.63 / 0.67 accuracy (majority baselines 0.53 / 0.74) and *verifies
fewer steps than always running Python* (0.30 vs 0.40 held out), because it under-routes to the
cheap verifier. The economics make the flat formulation wrong: always-Python already reaches 0.40
of the oracle's 0.47 (held) and 0.21 of 0.26 (hard) at negligible cost, and Lean's whole added
value is the 5–7 % of steps Python misses but Lean proves, at ~20× the cost per call.

**Two-stage, cost-aware router** (`router_eval2.py`): run Python on every step; among the steps it
did not verify, a probe predicts "Lean would prove this" and only the top b % of all steps go to
Lean. Fit on 2,733 python-unverified fit steps (18.7 % Lean-provable):

| stage-2 probe | set | AUC (lean \| ¬python) | verified share: Python only → +Lean on 5 % / 10 % / 20 % / 30 % of steps | oracle | precision of the 10 % sent |
|---|---|---|---|---|---|
| Goedel residual (facts) | held | 0.646 | 0.402 → 0.415 / 0.423 / 0.437 / 0.449 | 0.469 | 0.21 (random 0.11) |
| Goedel residual (facts) | hard trees | **0.811** | 0.205 → 0.224 / **0.234** / 0.243 / 0.246 | 0.257 | **0.29** (random 0.07) |
| generator state | held | 0.652 | 0.402 → 0.418 / 0.426 / 0.438 / 0.449 | 0.469 | 0.23 |
| generator state | hard trees | 0.653 | 0.205 → 0.215 / 0.221 / 0.228 / 0.234 | 0.257 | 0.17 |

Reading: routing works where it matters most. On hard problems the formalizer's state tells
which Python-unverified steps Lean can close (AUC 0.81): sending 10 % of steps to Lean recovers
+2.9 of the 5.2 points between Python-only and the oracle, four times what random routing would
(0.7). On easy problems the signal is weaker (0.65) and the Lean margin smaller. The probe is
saved as `probe/probes/router_lean_given_nopython.npz` for the routed decode (§13.5).
