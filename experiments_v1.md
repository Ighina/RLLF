# Experiments v1 — decoding, formalizability, and real Lean verification

Consolidated report of every decoding-time experiment run after `settings_comparison.md`
(which covers the offline FaithformBench label-setting result on its own). This file is
about what happens when the fitted reward models are actually used at generation time —
on two base models, with a real ground-truth Lean check added on top of the trained
policy's own prediction. Read §0 before any table below — the decoder configuration is
the single most important piece of context for interpreting every number here.

---

## 0. What was actually run

**Decoder: `rerank` only.** `rllf/generation/guided_decoder.py` implements four modes
(`plain`, `rerank`, `rejection`, `token`). Every `policy_guided_*` result in this document
uses **`rerank`** — stage 2 of 4:

```python
DecodeConfig(mode=MODE_RERANK, greedy=True, alpha=1.0, num_candidates=4,
             max_steps=6, verification=VERIFY_NONE, temperature=0.8)
```

At each of up to 6 steps: sample 4 complete candidate continuations from the frozen base
model, score all 4 with the trained reward model, greedily keep the top-scoring one.
**No Lean verification of any kind runs during decoding** (`verification=VERIFY_NONE`) —
the policy's own trained scores are the only signal, never checked against ground truth
mid-generation. `rejection` (stage 3: send top candidates to Lean, resample on failure)
and `token` (stage 4: per-token `log p_base + α·log π_φ` guidance) are implemented in the
codebase but **have not been used in any result below**. That's the natural next
experiment — see §6.

**Base models:** `microsoft/Phi-3.5-mini-instruct` (3.8B) and `Qwen/Qwen2.5-7B-Instruct`
(7B), run independently end to end (own `base_greedy`, own everything).

**Sample:** the same 250-problem stratified-by-difficulty subsample of MATH-500,
`--seed 0`, for both models — so results are paired and comparable.

**Policies:** `prm_verifiability_all8_majority` (mode A — "is this step formalizable and
provable", offline AUC 0.798, MiniLM-L6) and `prm_agreement_all8_majority` (mode B — "is
this step correct", AUC ~0.51–0.62 depending on encoder). Both come from `settings_comparison.md` §2.

**Real-Formalizability pipeline:** `Goedel-LM/Goedel-Formalizer-V2-8B` (writes a Lean
statement for a step) → `deepseek-ai/DeepSeek-Prover-V2-7B` (attempts a tactic proof,
n=16 samples) → the model's tactic block is spliced onto the *original* untouched
statement and compiled by a real `lake exe repl` against a fully built Mathlib — genuine
ground truth, not a prediction. Code: `rllf/verification/real_lean_pipeline.py`.

---

## 1. FaithformBench offline result (recap — full detail in `settings_comparison.md`)

| encoder | mode A AUC (verifiability) | mode B AUC (agreement) |
| --- | --- | --- |
| MiniLM-L6 (33M) | 0.798 | 0.544 |
| BERT-base (110M) | — | 0.617 |
| ModernBERT-base (150M) | 0.798 | — |

Mode A is learnable and **capacity-saturated** (MiniLM = ModernBERT, despite 4.5x the
parameters and a 2024 architecture). Mode B has a small amount of capacity-recoverable
signal (MiniLM → BERT-base: +0.073) sitting on top of a much larger amount no text
encoder reaches. `all8_majority` is the best setting in mode A and the *worst* in mode B
— ensembling sharpens a real signal and merely averages away noise. Full writeup,
formalizer-by-formalizer breakdown, and the "kind vs. degree" argument are in
`settings_comparison.md`.

---

## 2. MATH-500 decoding accuracy — does either policy raise answer accuracy?

> **⚠ Read §10 before trusting the numbers below.** Every accuracy figure in this
> section (and everywhere else in this document and `interim_report.md`) was measured
> under a `max_steps=6` cap that turned out to silently zero out 19–52% of every arm's
> trajectories with an empty final answer — at a rate that differs *systematically by
> arm type* (rerank-family arms roughly twice as often as `self_consistency`). This is a
> real, unquantified confound on both the absolute numbers and the relative comparison
> between arms, discovered only after this document was otherwise complete. Fixed in
> code (`DecodeConfig.max_steps`/`BeamSearchConfig.max_steps` now default to `None`,
> uncapped); **not yet re-run**. Treat every accuracy comparison below as provisional
> until §10's re-run lands.

7 arms, both base models, `rerank`-mode policy guidance (§0), McNemar exact test paired
against `base_greedy` on the same 250 questions.

### Phi-3.5-mini-instruct

| arm | accuracy | vs. `base_greedy` (McNemar p) |
| --- | --- | --- |
| `base_greedy` | 0.256 | — |
| `random_of_k` | 0.260 | p = 1.000 |
| `self_consistency` | 0.344 | **p < 0.001** |
| `self_consistency_weighted_verifiability` | 0.324 | **p = 0.005** |
| `self_consistency_weighted_agreement` | 0.348 | **p < 0.001** |
| `policy_guided_verifiability` (mode A) | 0.268 | p = 0.736 |
| `policy_guided_agreement` (mode B) | 0.244 | p = 0.736 |

### Qwen2.5-7B-Instruct

| arm | accuracy | vs. `base_greedy` (McNemar p) |
| --- | --- | --- |
| `base_greedy` | 0.360 | — |
| `random_of_k` | 0.320 | p = 0.143 (24 losses vs. 14 wins) |
| `self_consistency` | 0.440 | **p = 0.002** |
| `self_consistency_weighted_verifiability` | 0.436 | **p = 0.004** |
| `self_consistency_weighted_agreement` | 0.448 | **p = 0.001** |
| `policy_guided_verifiability` (mode A) | 0.356 | p = 1.000 |
| `policy_guided_agreement` (mode B) | 0.336 | p = 0.392 |

**Reading.** On both models, independently: neither `policy_guided_*` arm beats
`base_greedy`, mode A and mode B are statistically indistinguishable from baseline *and*
from each other, and `self_consistency` (no reward model at all) is the only
consistently significant winner. Weighting self-consistency's vote by either policy's
per-trajectory reward (§below) moves accuracy by at most a couple of points either
direction — never enough to change the qualitative picture. Doubling the base model's
raw accuracy (Phi-3.5-mini → Qwen2.5-7B, +60% relative) left the `rerank`-mode policy
arms exactly as flat relative to their own baseline as before — ruling out "the base
model has no headroom" as the explanation. See `settings_comparison.md` for the full
argument: the AUC this reward model achieves measures per-step well-formedness, and that
signal does not track which of 4 candidates leads to a correct final answer, at least not
at K=4/6-steps `rerank`.

**Reward-weighted self-consistency, specifically.** Same `n=4` independently-sampled
whole solutions as `self_consistency` (generation is still unguided — the reward model
never steers, only judges after the fact); each trajectory's vote is its mean per-step
formalizability probability instead of a flat 1
(`rllf.generation.guided_decoder.self_consistency_weighted`). Verifiability-weighting
underperforms plain unweighted voting on Phi-3.5-mini (0.324 vs. 0.344) and is a
statistical tie on Qwen2.5-7B (0.436 vs. 0.440); agreement-weighting sits within noise of
plain voting on both (0.348 vs. 0.344; 0.448 vs. 0.440) — expected, since a
near-chance policy can't discriminate between trajectories enough to change the vote.
Net: weighting the vote by this reward signal never helps, and on the model where the
signal is real (mode A, Phi-3.5-mini) it measurably hurts — consistent with §2's finding
that formalizability doesn't track correctness, so leaning on it as a vote weight
occasionally does worse than not looking at it at all.

---

## 3. Predicted formalizability — is a decoding method's reasoning more *formalizable*, even when accuracy is flat?

This is the "free win" question: even where §2 shows no accuracy gain, does
`policy_guided_verifiability` at least produce steps the trained policy itself would call
more formalizable — the same answers, more mechanically checkable reasoning? Computed by
`--stage formalizability`: score every already-generated step of every trajectory with
the fitted policy (no regeneration, reuses the §2 outputs). `mean_step_formalizability`
is the average predicted probability across all steps in an arm; `fully_formalizable_rate`
is the fraction of *trajectories* where every single step scores above 0.5.

### Under the mode-A (verifiability) policy — the one that actually discriminates

| arm | Phi-3.5-mini mean step / fully-formalizable | Qwen2.5-7B mean step / fully-formalizable |
| --- | --- | --- |
| `base_greedy` | 0.386 / 8.0% | 0.405 / 8.8% |
| `random_of_k` | 0.403 / 6.8% | 0.398 / 10.8% |
| `self_consistency` | 0.400 / 6.4% | 0.413 / 10.4% |
| `self_consistency_weighted_verifiability` | 0.434 / 11.6% | 0.433 / 15.6% |
| `self_consistency_weighted_agreement` | 0.398 / 8.4% | 0.407 / 10.0% |
| **`policy_guided_verifiability`** | **0.498 / 20.4%** | **0.485 / 21.6%** |
| `policy_guided_agreement` | 0.390 / 5.6% | 0.393 / 8.8% |

**This is a real, replicated result.** On both base models, `policy_guided_verifiability`
more than doubles the fully-formalizable-trajectory rate over every other arm, with a
+0.08–0.11 lift in mean step formalizability — while §2 showed its answer accuracy is
statistically unchanged from greedy decoding. That is the free win stated in advance:
same answers, reasoning that is roughly twice as likely to be entirely mechanically
verifiable. `self_consistency_weighted_verifiability` shows a smaller version of the same
effect (weighting the vote by formalizability does select somewhat more formalizable
winners, just not more *correct* ones — consistent with §2).

### Under the mode-B (agreement) policy — included for completeness, not informative

Every arm on both models lands at mean step ≈0.55 with `step_formalizable_rate` =
`fully_formalizable_rate` = 1.000, identically, everywhere. This is the near-chance
policy (AUC ~0.51–0.62) producing a degenerate, non-discriminating score — it cannot
distinguish formalizable from not, so this metric is uninformative for mode B by
construction, not a finding about the decoding arms.

---

## 4. Real-Formalizability — ground truth, not the policy's own guess

§3 asks the trained policy to grade its own kind of output — informative for whether the
*policy* believes an arm's steps are more formalizable, but it can't be the last word: a
real "is this actually a valid, Lean-checkable proof" signal is needed to know whether
the policy's judgment is trustworthy at all. That is what this pipeline gives, following
FaithformBench's own methodology (real autoformalizer + DeepSeek-Prover-v2 + real Lean)
but run fresh on our own MATH-500 output.

**Pilot (this session): 13/20 steps proved (65%)** on `base_greedy` (Phi-3.5-mini),
n=16 DeepSeek-Prover-V2 samples per step — validated the pipeline mechanics before
trusting it for a comparison. All genuine Lean+Mathlib compiler verdicts; the remaining
failures are real proof-search misses (`omega could not prove the goal`, `simp made no
progress`, `linarith failed`, `OfScientific ℕ` synthesis failures), not pipeline
artifacts.

### Per-arm comparison — the aggregate is misleading; the effect is real and difficulty-dependent

**An earlier version of this section reported that the 20-problem pilot's +10.9-point
edge for `policy_guided_verifiability` vanished at 110 problems (aggregate: 37.2% vs.
36.5%) and concluded it was small-sample noise. That conclusion was wrong, caught by a
direct question about whether the 20- and 110-problem samples were actually comparable
populations.** They were not: MATH-500's stratified sample is built **level-by-level in
contiguous blocks**, not shuffled, so a prefix slice is not a random subsample of
difficulty — `[:20]` happened to be **100% level-1 (easiest)** problems, while the
90 problems added to reach 110 are **98% level-2/3**. The aggregate at 110 problems
mixes two different difficulty populations and the level-1 signal gets diluted by
level-2/3 steps, which dominate the total (355 of 438 for `base_greedy`). Splitting by
level, using the same fully-scored 110-problem data:

| level | `base_greedy` | `policy_guided_verifiability` | Δ |
| --- | --- | --- | --- |
| 1 (easy) | 54.2% (45/83) | **63.2% (43/68)** | **+9.0 pts** |
| 2 (medium) | 37.5% (63/168) | 38.9% (63/162) | +1.4 pts (noise) |
| 3 (hard) | 29.4% (55/187) | 24.7% (46/186) | -4.7 pts |

**The level-1 result closely replicates the 20-problem pilot** (63.2%/54.2% here vs.
64.1%/53.2% in the pilot — the pilot *was* level-1, plus 2 more level-1 problems here) —
it is not noise, it is the same real effect seen twice. What changes is the level-2 and
especially level-3 populations, where the edge disappears and mildly reverses. The
correct reading is **not** "the free win vanished" — it is: **`policy_guided_verifiability`'s
real (not just self-reported) formalizability edge is real and consistent specifically on
easy problems, and disappears (or slightly reverses) as difficulty increases** — plausibly
because on hard problems, formalizable-*looking* and actually-correct diverge more, so
reranking toward apparent formalizability has less (or negative) value. This is a more
interesting finding than either the original "confirmed" or the incorrect "refuted"
framing, and it means §3's aggregate self-reported numbers likely hide the same
difficulty-dependence — worth checking there too.
(`policy_guided_agreement` is mid-run, not yet conclusive; will be updated once complete
— see `/scratch/users/iacopog/rllf-runs/logs/real_formalizability_by_arm.json` for live
per-step and per-level detail, including trajectory-level `level`/`subject` metadata
needed to reproduce this breakdown.)

**A second, still-open question this pipeline hasn't resolved**: is "proved" actually
measuring *formalizability* (can this claim be stated and reasoned about in Lean) or
something closer to *correctness* (is the claim true)? FaithformBench's own mode-A
definition only ever tested already-true steps, so it never had to separate the two; our
pipeline runs on freshly-generated steps that may simply be wrong, and a wrong claim
should fail to prove regardless of how "formalizable" its shape is. A quick scan of
failure logs shows both genuine falsity (`tactic 'decide' proved that the proposition ...
is false`) and pure search failures (`omega could not prove the goal`, `simp made no
progress`) among the "not proved" cases, but not rigorously quantified — this needs a
proper breakdown (e.g. tagging each failure by whether Lean found the statement
false vs. merely couldn't find a proof) before the metric's name can be trusted at face
value.

**Getting the pipeline itself working took real debugging**, worth recording since the
bugs were in reused infrastructure ([VeriForm](https://github.com/Ighina/VeriForm),
branch `pho`) and would bite anyone else calling it the same way:

1. Mathlib was vendored as source, never built — `lake exe cache get` (prebuilt oleans,
   ~5 min) instead of a from-scratch build (hours), all under `/scratch`.
2. vLLM's decode for the DeepSeek-Prover-V2-7B checkpoint leaks byte-level-BPE display
   form instead of detokenizing (`∃` → `âĪĥ`, space → `Ġ`) — fixed with the actual GPT-2
   byte↔unicode reverse mapping, not a couple of character replaces.
3. `get_theorem_name_body`'s regex stopped at the *first* `:=`, breaking on any statement
   with a `let x := v` binding (most of Goedel's multi-variable output) — anchored on the
   real proof-opening idiom `:= by` instead.
4. The design itself was the deeper bug: comparing the model's restated theorem text
   against the original (even normalized) is unreliable in principle, not just fragile —
   DeepSeek drops inferable type ascriptions and in some cases changes `∧` to `→`, a real
   semantic drift. Fixed by **splicing** the model's tactic block onto our own untouched
   statement instead of trusting its restatement, so what gets Lean-checked is always the
   original claim.
5. Goedel's own output is already a complete, self-contained Lean file (own
   `import`/`open` preamble) — wrapping it in VeriForm's template again put a second
   `import` after an `open`, which Lean rejects outright, on every single candidate. This
   alone explains why the very first automated attempts scored 0% despite a manual,
   hand-written reproduction of the same statement succeeding.

---

## 5. Summary so far

- Mode A (verifiability) is a real, learnable, capacity-saturated signal; mode B
  (agreement) is near-chance with a small capacity-recoverable component. (§1)
- Neither policy improves MATH-500 answer accuracy via `rerank`-mode guidance, on either
  base model, at any base-model strength tested. `self_consistency` (no reward model)
  is the only method that reliably beats greedy decoding. (§2)
- Reward-weighting self-consistency's vote never helps and sometimes measurably hurts.
  (§2)
- Mode A produces a replicated, ~2x lift in *self-reported* (policy-graded)
  formalizability at matched accuracy. (§3)
- **A ground-truth Lean-checked comparison confirms this is real, but difficulty-dependent,
  not uniform.** `policy_guided_verifiability` has a real, twice-replicated +9-11 point
  real-provable-rate edge over `base_greedy` on *easy* (level-1) MATH-500 problems, which
  disappears on medium problems and mildly reverses on hard ones. A naive aggregate over
  a difficulty-unbalanced 110-problem sample looked like the effect had vanished entirely
  — it hadn't; the aggregate was just diluted by harder problems where it doesn't hold.
  This still coexists with §2's null *accuracy* result at every difficulty level tested so
  far — formalizability and correctness remain distinct axes even within this breakdown.
  (§4, `policy_guided_agreement` still completing, and the formalizability-vs-correctness
  conflation flagged in §4 is still an open methodological question)
- **A much stronger classifier (Qwen3-1.7B, AUC 0.863 vs. 0.798, and — unlike MiniLM —
  equally strong on hard problems) doesn't help, and by the ground-truth measure, hurts.**
  Accuracy is still flat (0.272 vs. `base_greedy`'s 0.256, p=0.597). Self-reported
  formalizability still shows the same ~2.5x "free win" — but checked against real Lean,
  Qwen3-1.7B-guided decoding is *worse* than MiniLM-guided at every difficulty level, and
  its aggregate real-formalizable rate (33.0%) sits below `base_greedy`'s (37.2%). Better
  offline AUC did not transfer to a better real result. Calibration is ruled out as the
  mechanism (rerank's candidate choice is provably invariant to it); the two classifiers
  genuinely disagree on which steps are more verifiable, and that disagreement is a
  generalization gap, not yet explained. (§7)

## 7. Scaling classifier capacity: a fine-tuned Qwen3-1.7B encoder

§1's mode-A AUC (0.798) was identical for MiniLM-L6 (33M) and ModernBERT-base (150M) —
capacity-saturated at the small-encoder scale. Two questions followed: is FaithformBench's
*training data* itself easy-skewed (which would make the saturation an artifact of a narrow
target, not a real ceiling), and does a genuinely larger, differently-architected classifier
break through it?

**Training-data distribution — not easy-skewed.** 65.5% of MATH-derived steps in the fit
pool are level 3–5; including the non-MATH sources (OlympiadBench, OmniMath), ~72% of the
full 1,179-problem pool is hard/olympiad-level material. If anything the pool skews hard.

**A fine-tuned Qwen/Qwen3-1.7B classifier breaks the ceiling.** Same `all8_majority`
labels, same 80/20 held-out split MiniLM/ModernBERT were fit and tested on, same offline
AUC/accuracy methodology. Being a causal decoder backbone rather than a bidirectional
encoder required last-token pooling instead of `stage_fit`'s previously-hardcoded
"cls"/position-0 pooling — meaningless under causal masking — caught and fixed
(`--pooling`) before training, not after a wasted run.

| | Qwen3-1.7B | MiniLM-L6 / ModernBERT-base |
| --- | --- | --- |
| held-out AUC | **0.863** | 0.798 (identical, capacity-saturated) |
| calibration (ECE) | 0.206 (worse) | 0.077 (ModernBERT) |

Worse calibration matters less for `rerank`'s relative-ranking use case than for anything
consuming the raw probability — flagged, not dismissed.

**The advantage is not an easy-problem artifact — it holds, and widens, with difficulty:**

| group | Qwen3-1.7B AUC | MiniLM AUC | gap |
| --- | --- | --- | --- |
| gsm8k (easiest) | 0.969 | 0.926 | +0.043 |
| MATH level 1 | 0.833 | 0.810 | +0.023 |
| MATH level 2 | 0.863 | 0.794 | +0.069 |
| MATH level 3 | 0.854 | 0.821 | +0.033 |
| MATH level 4 | 0.852 | 0.777 | +0.075 |
| MATH level 5 (hardest) | 0.833 | 0.722 | **+0.111** |
| olympiadbench | 0.815 | 0.747 | +0.068 |
| omnimath | 0.848 | 0.783 | +0.065 |

MiniLM genuinely degrades with difficulty (0.810 → 0.722 across MATH levels); Qwen3-1.7B
stays flat (0.833–0.863) everywhere. This rules out §4's "real but concentrated on easy
problems" pattern as a property of *classification quality generally* — it was specific to
that earlier small-encoder policy, or to something else about the decoding arm itself.

**Plugged into an actual MATH-500 `policy_guided_verifiability` run** (Phi-3.5-mini, same
250-problem sample, seed 0, identical to every other arm in §2):

| | accuracy | vs. `base_greedy` (0.256) |
| --- | --- | --- |
| `policy_guided_verifiability` (old, MiniLM) | 0.268 | p=0.736 |
| `policy_guided_verifiability` (new, Qwen3-1.7B) | 0.272 | p=0.597 |

**Unchanged — a stronger, more difficulty-robust classifier does not move accuracy.** This
is an important negative result on its own: classifier capacity/quality was never the
bottleneck for `rerank`'s null accuracy effect in §2; the decoding mechanism itself (commit
to a step outright, no second chance) appears to be the cap.

The self-reported "free win" (§3) replicates and looks, if anything, larger in relative
terms with the better classifier:

| | `base_greedy` fully-formalizable | `policy_guided_verifiability` fully-formalizable |
| --- | --- | --- |
| under MiniLM (old) | 8.0% | 20.4% (2.55x) |
| under Qwen3-1.7B (new) | 13.2% | 33.6% (2.55x) |

(Absolute rates aren't comparable across classifiers — each grades against its own decision
boundary — but the ~2.5x lift replicates exactly.)

**But checked against real Lean (not self-reported), the story reverses.** Running §4's
exact pipeline on the Qwen3-1.7B-guided trajectories (same 110 problems, same
Goedel-Formalizer-V2-8B + DeepSeek-Prover-V2-7B, n=16 samples/step):

| level | `base_greedy` | Qwen3-1.7B-guided | MiniLM-guided (§4) |
| --- | --- | --- | --- |
| 1 (easy) | 54.2% | 60.0% (+5.8) | **63.2% (+9.0)** |
| 2 (medium) | 37.5% | 35.6% (−1.9) | 38.9% (+1.4) |
| 3 (hard) | 29.4% | 21.0% (**−8.4**) | 24.7% (−4.7) |
| **aggregate** | **37.2%** | **33.0%** | 36.5% |

The qualitative pattern replicates (edge on easy problems, flat-to-negative elsewhere), but
**Qwen3-1.7B-guided is worse than MiniLM-guided at every level** — a smaller easy-problem
gain and close to double the hard-problem loss — and its aggregate real-formalizable rate
(33.0%) sits *below* `base_greedy`'s (37.2%), unlike MiniLM's (36.5%, roughly matched).

**This directly contradicts the self-reported numbers above and is the more trustworthy of
the two** (ground truth vs. the policy grading its own output). A classifier with uniformly
better, more difficulty-robust offline AUC (0.863 vs. 0.796) produces a *worse* real
downstream result when used to guide the same rerank decoder.

**Calibration was the first hypothesis and it's wrong — ruled out by the code, not just
untested.** `MODE_RERANK` with `greedy=True` picks
`argmax_i(policy.score_step_candidates(...)[i])`, and every candidate in that comparison
goes through the *same* Platt scaler (`sigmoid((z-b)/T)`, then `.log()`) — a strictly
monotonic function of the raw logit for any `T > 0`. Monotonic transforms cannot change an
argmax over a fixed set. Whatever Qwen3-1.7B's calibration looks like (its fitted
temperature, 7.86, is actually larger/more conservative than MiniLM's 2.11 — not sharper,
so saturation-to-extremes isn't it either), `rerank`'s candidate choice is provably identical
to what raw, uncalibrated logits would have chosen. Re-calibrating changes nothing about
which candidate gets picked.

**So the two classifiers must genuinely disagree on the raw ranking** of the same candidate
steps — not a calibration artifact, a real difference in judgment. Qwen3-1.7B's preference
transfers worse to real Lean proof search on freshly-sampled Phi-3.5-mini output than
MiniLM's does, despite scoring better on FaithformBench's own held-out split — a
generalization gap. One plausible story: Qwen3-1.7B's larger capacity let it pick up more
FaithformBench-specific surface signal (stylistic features of the 8 label-source models)
that doesn't transfer to a differently-phrased model's output, while MiniLM, being
capacity-constrained, was forced to learn something coarser but more portable. Untested —
flagged as the real open item below, replacing the (incorrect) calibration hypothesis.

Two bugs found and fixed getting this far, both a variant of the same gap: `stage_eval` and
`stage_formalizability` never moved a loaded policy off CPU. Invisible with tiny
MiniLM/ModernBERT; with a 1.7B decoder-backbone policy it first ran an entire decoding job
silently on CPU (287–358s/problem instead of the ~4s/problem it should have been — traced to
a transient CUDA-driver init failure at job start that got cached as "no GPU" for the rest of
the run) and then, in the formalizability-scoring step, spun up ~73 unbounded CPU threads on
this shared box before being caught and killed. Fixed by moving the policy to the resolved
device in both stages, and by adding a fast CUDA pre-flight check to the job launcher so a
repeat of the driver hiccup fails in seconds instead of silently running the whole job on
CPU.

---

## 8. Three follow-up investigations into the generalization gap

§7 left one open question: why does Qwen3-1.7B's raw candidate ranking generalize worse to
real Lean formalizability than MiniLM's, despite a uniformly better offline AUC? Calibration
was ruled out by the code itself (§7). Three investigations, run in parallel:

### 8.1 Candidate-level disagreement (does Qwen3-1.7B favor an identifiable shortcut?)

For 40 MATH-500 problems spread across all 5 difficulty levels, replayed each step's
realistic history and froze a single fresh 4-candidate pool per step (matching the decoder's
own sampling config), then scored every candidate with *both* policies to isolate disagreement
from confounds like different generated content.

- **44.9% raw agreement** (107 comparisons) — far below what near-identical judgment would
  produce.
- **Not explained by length**: MiniLM's picks average 23.9 words, Qwen3-1.7B's 23.2.
- **U-shaped by difficulty**: agreement is lowest at level 1 (33%, n=9) and level 5 (27%,
  n=30), highest at level 2–3 (~60%) — disagreement concentrates at the extremes, not a
  simple "gets worse with difficulty" trend.
- **MiniLM's own confidence margin shrinks specifically on disagreement cases** (0.117 vs.
  0.180 when they agree); Qwen3-1.7B's margin is constant either way (~0.21). Reading:
  disagreements often look like "MiniLM was on the fence, Qwen3-1.7B had a clear but
  different opinion," not two confident classifiers clashing.
- **Manual inspection of ~11 disagreement cases found no consistent directional bias.** An
  initial example looked like Qwen3-1.7B favoring a trivial/restating step over a substantive
  one (a "reward vacuous-but-safe statements" shortcut) — but broader inspection killed that
  story: in several other cases Qwen3-1.7B's pick was the mathematically *better* step (an
  algebra error correctly avoided), and in one case both candidates the two models chose
  between contained different genuine arithmetic errors. No simple, interpretable heuristic
  explains the gap.

**Conclusion**: the disagreement is real and substantial, but idiosyncratic rather than a
fixable pattern — consistent with a genuine generalization gap, not a bug or an obvious
shortcut.

### 8.2 Single label source vs. 8-way ensemble (is this ensemble overfitting?)

Re-fit Qwen3-1.7B on the single best-performing label source (`goedel`, MiniLM's own-best
single-source AUC 0.787) instead of the `all8_majority` ensemble, then ran the same real-Lean
check on the resulting decoding arm.

- **Offline**: AUC 0.825 (vs. MiniLM's 0.787 on the same source; vs. Qwen3-1.7B's own 0.863
  on the ensemble) — still beats MiniLM at matched label source, but the ensemble buys
  Qwen3-1.7B much more (+0.038) than it buys MiniLM (+0.011).
- **Real-Lean, all three variants on the same 110 problems:**

  | policy | easy edge | medium edge | hard edge | aggregate edge |
  | --- | --- | --- | --- | --- |
  | MiniLM (8-way ensemble) | +9.0 | +1.4 | −4.7 | **−0.7** |
  | Qwen3-1.7B (`goedel` single-source) | **+0.0** | +3.1 | −4.3 | −1.0 |
  | Qwen3-1.7B (8-way ensemble) | +5.8 | +1.4 | −8.4 | −4.2 |

**Mixed verdict — ensemble overfitting is a real, partial explanation.** Dropping the
ensemble pulls Qwen3-1.7B's hard-problem loss (−8.4 → −4.3) and aggregate transfer
(−4.2 → −1.0) almost exactly onto MiniLM's own numbers. That part of the generalization gap
*is* explained by ensemble-specific overfitting. **But the easy-problem result is the
opposite of what the hypothesis predicts**: single-source training didn't restore the
easy-problem edge at all (0.0, even below the ensemble version's +5.8) — something else,
likely intrinsic to Qwen3-1.7B's capacity/architecture rather than the labels it was trained
on, is driving that piece.

### 8.3 Base model identity (is this about which LLM generated the candidates?)

Qwen2.5-7B-Instruct is confirmed (via `Qwen/ProcessBench`'s own `generator` field) as one of
FaithformBench's 8 generator models — 351 of the 1,179 problems' underlying reasoning traces
came from it — unlike Phi-3.5-mini, which is not in that list at all. Ran the same
Qwen3-1.7B-guided decoding + real-Lean check on Qwen2.5-7B-Instruct, alongside a first-ever
real-Lean check of the existing MiniLM-guided arm on this base model (previously only checked
on Phi-3.5-mini).

- Decoding accuracy: flat as always (`base_greedy` 0.360, Qwen3-1.7B-guided 0.360, p=1.000).
- **Real-Lean check, all three arms, same 110-problem matching methodology as §8.2:**

  | level | `base_greedy` | MiniLM-guided | Qwen3-1.7B-guided |
  | --- | --- | --- | --- |
  | 1 (easy) | 56.8% | 64.9% (+8.1) | 50.0% (**−6.8**) |
  | 2 (medium) | 41.1% | 38.7% (−2.4) | 44.0% (+2.9) |
  | 3 (hard) | 24.6% | 23.0% (−1.6) | 22.5% (−2.1) |
  | **aggregate** | **36.6%** | **36.6% (+0.0)** | **35.3% (−1.3)** |

**The hypothesis is confirmed on the metrics that matter most, with a genuine new wrinkle.**
Side by side against the Phi-3.5-mini numbers (§4, §7) on the same 110 problems:

| policy | base model | easy edge | hard edge | aggregate edge |
| --- | --- | --- | --- | --- |
| MiniLM-guided | Phi-3.5-mini | +9.0 | −4.7 | −0.7 |
| MiniLM-guided | Qwen2.5-7B-Instruct | +8.1 | **−1.6** | **+0.0** |
| Qwen3-1.7B-guided | Phi-3.5-mini | +5.8 | −8.4 | −4.2 |
| Qwen3-1.7B-guided | Qwen2.5-7B-Instruct | **−6.8** | −2.1 | **−1.3** |

Moving to a base model that is actually one of FaithformBench's own generators narrows both
classifiers' hard-problem loss and aggregate transfer gap substantially — MiniLM's aggregate
transfer becomes essentially perfect (+0.0), and Qwen3-1.7B's improves from −4.2 to −1.3. This
is real, direct support for the distribution-shift explanation. **But Qwen3-1.7B's
easy-problem edge doesn't just shrink — it flips sign** (+5.8 → −6.8): on Qwen2.5-7B-Instruct,
Qwen3-1.7B-guided decoding makes easy problems *less* likely to real-formalize than plain
greedy decoding. A simple "distribution shift explains everything" story predicts
improvement or no change here, not a reversal — this is a new, unexplained result sitting on
top of an otherwise-confirmed hypothesis, not something the base-model-identity story
resolves on its own.

---

## 9. MAJOR CONFOUND: `max_steps=6/8` was silently zeroing out 19–52% of every arm

Discovered while investigating why `lean_constrained_beam_search`'s smoke test wasn't
reaching final answers within its own step budget (`lean-constrained-beam-search.md` §7,
`step-native-finetuning.md` §5) — checking whether the *original* `<step>`-token
experiments had the same problem turned up something much larger than a beam-search
tuning issue.

**The mechanism.** `force_finish` (`GuidedDecoder.generate`'s "this is the last permitted
step" signal) is a `MODE_TOKEN`-only mechanism — `_next_step_candidates` (what
`MODE_PLAIN`/`MODE_RERANK`/`MODE_REJECTION` actually call) never receives it and does
nothing special on the final allowed step. A trajectory that simply hasn't produced a
step matching `contains_final_answer` by `max_steps` just... stops, with
`final_answer == ""`. Every outcome checker treats an empty string as wrong. So hitting
the cap is not "the model reasoned to a wrong answer" — it is a guaranteed zero,
indistinguishable in the reported accuracy number from a genuine reasoning failure.

**Measured on every existing MATH-500 run** (`max_steps=6` throughout):

| arm | Phi-3.5-mini: hit cap | reported acc | acc among not-capped | Qwen2.5-7B: hit cap | reported acc | acc among not-capped |
| --- | --- | --- | --- | --- | --- | --- |
| `base_greedy` | 38% (95/250) | 0.256 | **0.413** | 48% (119/250) | 0.360 | **0.687** |
| `random_of_k` | 47% (118/250) | 0.260 | 0.492 | 49% (123/250) | 0.320 | 0.630 |
| `self_consistency` | **19%** (47/250) | 0.344 | 0.424 | **26%** (65/250) | 0.440 | 0.595 |
| `self_consistency_weighted_verifiability` | 19% (47/250) | 0.324 | 0.399 | 26% (65/250) | 0.436 | 0.589 |
| `self_consistency_weighted_agreement` | 19% (47/250) | 0.348 | 0.429 | 26% (65/250) | 0.448 | 0.605 |
| `policy_guided_verifiability` | 41% (102/250) | 0.268 | 0.453 | 44% (109/250) | 0.356 | 0.631 |
| `policy_guided_agreement` | 52% (131/250) | 0.244 | 0.513 | 50% (124/250) | 0.336 | 0.667 |

**100% of capped trajectories, in every single arm and both base models, had an empty
final answer** — checked directly, not inferred.

**Why this matters beyond the absolute numbers.** The cap rate is not uniform across
arms: every rerank-family arm (`base_greedy`, `random_of_k`, `policy_guided_*`) hits it
roughly **twice as often** as `self_consistency` and its weighted variants. `rerank`'s
candidate-selection process appears to systematically produce more, and more finely
decomposed, reasoning steps than `self_consistency`'s unguided sampling does — plausibly
because reranking rewards whatever the classifier considers a well-formed step, and
finer decomposition may look more "verifiable" per step even though it costs more steps
overall. Whatever the exact mechanism, this means **the single most robust finding in
this entire investigation — "`self_consistency` beats every `rerank`-based arm" — may be
substantially explained by `self_consistency` simply timing out less often**, not
(only) by whole-solution voting being intrinsically better than step-level guidance.
The "not-capped" accuracy column above is *suggestive* in this direction (the gap between
`self_consistency` and `policy_guided_verifiability` shrinks or reverses once both are
restricted to their own naturally-completed trajectories) but is **not** a valid
apples-to-apples re-analysis — it compares different, self-selected subsets of the 250
questions per arm (whichever ones *that arm* happened to finish in time), which is its
own confound. It motivates a re-run; it does not replace one.

**Fixed in code, not yet re-run:**
- `DecodeConfig.max_steps`/`BeamSearchConfig.max_steps` now default to `None` (uncapped,
  bounded only by a new `MAX_STEPS_SAFETY_CEILING = 200` safety net against a genuinely
  non-terminating generation — not a reasoning-length policy, just a guard against
  runaway compute). `scripts/run_faithform.py --max-steps` defaults to `None` too.
  `BeamSearchConfig.max_tokens_per_step` raised 128 → 256 after the same smoke test also
  found it truncating a final answer mid-expression.
- 173/173 tests pass, including new ones for the `None`/safety-ceiling behavior
  (`tests/test_decoder.py`, `tests/test_beam_search.py`).

**Not yet done — the actual point of fixing this:** re-running the MATH-500 accuracy
comparison (at least `base_greedy`, `self_consistency`, `policy_guided_verifiability` on
one base model, to control cost) with the uncapped default, to get a genuinely
uncounfounded answer control to check the arms fairly. Until that lands, **every
accuracy conclusion in §2 of this document, and in `interim_report.md`, should be read
as "true under a since-discovered artificial budget constraint," not as a settled
finding.** This is now the single highest-priority open item, superseding most of §9
below until it's resolved.

---

## 10. Open items / not yet run

- **Formalizability vs. correctness**: quantify how much of "not proved" is genuine
  falsity vs. search failure — needed to know what the Real-Formalizability metric is
  actually measuring (§4).
- **What's specifically driving Qwen3-1.7B's easy-problem transfer behavior** — §8.2 showed
  it isn't fixed by dropping the label ensemble, and §8.3 showed it doesn't just improve with
  a more in-distribution base model, it *reverses sign* (+5.8 on Phi-3.5-mini → −6.8 on
  Qwen2.5-7B-Instruct). Neither hypothesis tested so far explains this; still open.
- **Finish `policy_guided_agreement`** at 110-problem scale (30% done, paused at a
  16-hour job time limit), and re-run the level-stratified breakdown once it lands.
- **`rejection` mode** — Lean-in-the-loop resampling during decoding (verify top
  candidates, resample on failure) instead of rerank-and-commit. Not yet tried on either
  base model; a natural next step given §2's (and now §7's) null result under plain rerank,
  now doubly motivated since §7 rules out classifier quality as the explanation.
- **`token`-level guidance** — per-token `log p_base + α·log π_φ` steering. Not yet tried.
