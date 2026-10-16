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

> **⚠ Superseded — see §9.** Every accuracy figure below was measured under a
> `max_steps=6` cap later found to silently zero out 19–52% of every arm's trajectories
> with an empty final answer, at a rate that differed systematically by arm (rerank-family
> arms roughly twice as often as `self_consistency`). §9 has the corrected, uncapped
> re-run for the three arms that mattered most: the absolute numbers below are all
> substantially understated, and `self_consistency`'s statistical significance over
> `base_greedy` does not fully survive correction (barely significant on Phi-3.5-mini,
> not significant at all on Qwen2.5-7B). The null result for `policy_guided_verifiability`
> is the one conclusion that does survive unchanged.

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

**Fixed in code:**
- `DecodeConfig.max_steps`/`BeamSearchConfig.max_steps` now default to `None` (uncapped,
  bounded only by a new `MAX_STEPS_SAFETY_CEILING = 200` safety net against a genuinely
  non-terminating generation — not a reasoning-length policy, just a guard against
  runaway compute). `scripts/run_faithform.py --max-steps` defaults to `None` too.
  `BeamSearchConfig.max_tokens_per_step` raised 128 → 256 after the same smoke test also
  found it truncating a final answer mid-expression.
- 173/173 tests pass, including new ones for the `None`/safety-ceiling behavior
  (`tests/test_decoder.py`, `tests/test_beam_search.py`).

**Re-run, both base models, `base_greedy`/`self_consistency`/`policy_guided_verifiability`
(the 3 arms that mattered most), uncapped:**

| arm | Phi-3.5-mini: capped → uncapped | Qwen2.5-7B: capped → uncapped |
| --- | --- | --- |
| `base_greedy` | 0.256 → **0.352** | 0.360 → **0.580** |
| `self_consistency` | 0.344 (p<0.001) → 0.408 (**p=0.049**) | 0.440 (p=0.002) → 0.620 (**p=0.121**) |
| `policy_guided_verifiability` | 0.268 (p=0.736) → 0.332 (p=0.568) | 0.356 (p=1.000) → 0.544 (p=0.163) |

A clean, consistent pattern across both models:

1. **The absolute accuracy of every arm was severely understated by the cap.** Uncapped
   Phi-3.5-mini's `base_greedy` nearly matches capped Qwen2.5-7B's `self_consistency`
   (0.352 vs 0.440) — the artificial budget was hiding most of both models' real capability.
2. **`self_consistency`'s advantage survives in raw accuracy but does not survive
   statistically.** On Phi-3.5-mini its significance drops from p<0.001 to a merely
   borderline p=0.049; on Qwen2.5-7B it drops from p=0.002 to **p=0.121 — not significant
   at all**. The single most confidently-stated finding in this whole investigation
   ("`self_consistency` reliably beats greedy decoding") does not fully survive the
   correction on the stronger model.
3. **The null result for reward-guided reranking is the one conclusion that is genuinely
   robust to this confound.** `policy_guided_verifiability` remains statistically
   indistinguishable from `base_greedy` on both models, both before and after
   correction — and is now numerically *below* baseline on both (losing more head-to-head
   matchups than it wins), rather than numerically above it as under the confounded
   measurement. Classifier-guided `rerank` genuinely does not improve accuracy; that part
   of the story was never a budget artifact.

Also notable: `policy_guided_verifiability` consistently needed the most steps to converge
of the three arms on Phi-3.5-mini (11.1 avg vs. 8.2/7.1) — consistent with §9's own
explanation for why it hit the old cap more often in the first place.

### 9.1 Second correction: the answer grader was rejecting correct LaTeX answers

Found while building the forked-tree data (`forked-tree-classifier-retraining.md` §3):
the composite outcome checker's rungs (exact match / numeric / sympy on Lean-normalised
text) only ever agreed on numbers or byte-identical strings, so `\(\frac{6}{5}\)` vs
`\frac65`, `76` vs `76^\circ`, `\sqrt{45}` vs `3\sqrt{5}`, `12π` vs `12\pi` were all graded
wrong. A new last rung, `MathEquivalenceChecker` (`rllf/verification/math_equivalence.py`),
plus balanced-brace `\boxed{}` extraction (`extract_boxed`), re-grades the stored uncapped
trajectories as follows (every newly-accepted answer in a random sample of 25 was checked
by eye and is genuinely correct; paired exact McNemar vs `base_greedy`):

| arm | Phi-3.5-mini: uncapped → re-graded | Qwen2.5-7B: uncapped → re-graded |
| --- | --- | --- |
| `base_greedy` | 0.352 → **0.376** | 0.580 → **0.664** |
| `self_consistency` | 0.408 (p=0.049) → 0.432 (**p=0.059**) | 0.620 (p=0.121) → 0.696 (**p=0.243**) |
| `policy_guided_verifiability` | 0.332 (p=0.568) → 0.364 (p=0.780) | 0.544 (p=0.163) → 0.640 (p=0.451) |

The Qwen2.5-7B arms move by +8–10 points (it answers in LaTeX far more often), Phi by
+2–3; the ordering and every conclusion in §9 are unchanged — `self_consistency`'s edge is
now non-significant on *both* models, and `rerank` remains indistinguishable from greedy.
All accuracy numbers elsewhere in this file predate the fix and are lower bounds; the
stored `.jsonl` trajectories re-grade in seconds on CPU when a table needs refreshing.

### 9.2 Third correction — the real-Lean pipeline was formalizing the *problem*, not the step

Found on the forked-tree run, the first real-Lean job that stored its Lean statements
alongside the steps. `render_step_claim` sends Goedel `Problem: … / Previous steps: … /
Step to verify: …`, and Goedel's fixed prompt asks it to "autoformalize the following
natural-language *problem statement*" — so, for most steps, it formalizes the problem.
Examples (forked-tree run): "Definition: cot θ = cos θ / sin θ" → `Real.cot (−π/3) = −√3/3`
(the answer); "In any triangle A+B+C=π" → the full determinant claim; "Let θ₁ = π/70" → the
entire reflection problem. Examples from the §8.3 cache, which also stored statements:
"Since sin y = 0.5, y = π/6" → `sin(arcsin 2/5 + arcsin 1/2)·… = 3/8` (the answer); "total
in cents is 12669" → `(…) % 100 = 69` (the answer); "f(−2) = 20−6+4" → `f(−2) = 18`.

Quantified on the four older caches that kept statements (a *lower* bound, since the
T=0.9 formalizer rephrases the same claim):

| cache | steps | share a verbatim Lean conclusion with another step of the same problem | (arm, problem) groups where *every* step got one identical statement |
|---|---|---|---|
| §4 `real_formalizability_by_arm` | 998 | 38% | 42 / 254 |
| §8.3 `real_formalizability_qwen7b_arms` | 1,271 | 35% | 47 / 330 |
| §8.1 `real_formalizability_qwen3_arm` | 443 | 33% | 16 / 110 |
| §8.2 `real_formalizability_qwen3_goedel_arm` | 456 | 35% | 16 / 110 |

Concrete-arithmetic steps (`26·25·10·9 = 58500`) were often formalized as themselves, so
step type did matter for a *subset* — which is what made the §10.1 step-type breakdown
look meaningful. But the "real formalization rate" of a trajectory in §4, §8 and §10 was to
a large degree *per-problem provability plus prover sampling noise* (n=16 proof attempts,
fresh formalization per step), with step content entering only sometimes. That is a
simpler explanation of §10.1 than the equation-density proxy: a classifier cannot rank
trajectories within a problem on labels whose within-problem variation is largely noise,
and labels that track problem difficulty will produce exactly the pooled-positive /
within-negative pattern reported there. §10.2's step-level AUCs against these labels are
likewise not what they were taken to be.

Not affected: the FaithformBench offline AUCs (§1, §7, §8 — the benchmark's own step-level
labels) and every accuracy number. Affected: every real-Lean rate in §4, §8.1–8.3, §10,
§10.1, §10.2 and `interim_report.md` §2.4–2.7, all of which now need re-labelling under a
step-level claim rendering. **Fix adopted:** `render_step_claim(style="facts")` — no
problem text, previous steps as numbered hypotheses, `Statement to formalize: <step>` —
chosen from six renderings probed on 37 steps (`forked-tree-classifier-retraining.md`
§7.2: every rendering that kept the problem text still fell back to it on some steps).
The old caches store the steps, so re-labelling them is a formalize+prove re-run, ~40 GPU-h
for all four. The earlier §7.2 regex bug (binder statements rejected) is real but
secondary to this.

---

### 9.4 Fourth correction — binder statements were never sent to the prover (re-run Oct 16)

`forked-tree-classifier-retraining.md` §7.2 found that `_THEOREM_RE` required a `:` right
after the theorem name, so every Goedel statement with explicit binders —
`theorem step_0 (x y : ℝ) (h : x + y = 10) : …` — was rejected as "no parseable theorem
statement" and recorded as a formalize-stage failure without ever reaching the prover.
That is 10–17% of the steps in every cache above. All 607 such entries across the five
legacy caches (§4's Phi arms, §8.3's Qwen2.5-7B arms, §7/§8.1's Qwen3 arms, §10's rerank
cache) were re-run with the fixed regex under the *legacy* claim rendering (so each cache
stays internally consistent; the §9.2 rendering change is a separate matter) and the
current prover settings (8 samples, 60 s, maxHeartbeats 400000 — slightly stricter than
the n=16/300 s the caches were built with). 10.8 GPU-h (`binder_rerun.py`,
`logs/binder_rerun.json`). **None still fails at the formalize stage**; **246 of 607 (40.5%)
prove** — a higher rate than the caches' prove-stage steps, as expected for the symbolic /
hypothesis-carrying steps the regex was dropping.

Corrected real-formalization rates (old → new; the ordering between arms is unchanged
everywhere):

| cache / arm | n | re-run | now proved | rate old → new | L1 | L2 | L3 |
|---|---|---|---|---|---|---|---|
| Phi-3.5 `base_greedy` (§4) | 438 | 69 | 20 | 0.372 → 0.418 | 0.54 → 0.54 | 0.38 → 0.46 | 0.30 → 0.33 |
| Phi-3.5 `policy_guided_verifiability` (§4) | 416 | 70 | 25 | 0.365 → 0.425 | 0.63 → 0.63 | 0.39 → 0.47 | 0.24 → 0.31 |
| Phi-3.5 `policy_guided_agreement` (§4, partial) | 144 | 15 | 8 | 0.493 → 0.549 | 0.56 → 0.58 | 0.37 → 0.49 | — |
| Qwen2.5-7B `base_greedy` (§8.3) | 434 | 70 | 29 | 0.366 → 0.433 | 0.57 → 0.62 | 0.41 → 0.48 | 0.25 → 0.32 |
| Qwen2.5-7B `policy_guided_verifiability` MiniLM (§8.3) | 415 | 73 | 28 | 0.366 → 0.434 | 0.65 → 0.69 | 0.39 → 0.46 | 0.23 → 0.30 |
| Qwen2.5-7B `policy_guided_verifiability` Qwen3-1.7B (§8.3) | 422 | 73 | 22 | 0.353 → 0.405 | 0.50 → 0.53 | 0.44 → 0.51 | 0.22 → 0.27 |
| Phi-3.5 `policy_guided_verifiability` Qwen3-1.7B (§7) | 443 | 79 | 22 | 0.330 → 0.379 | 0.60 → 0.63 | 0.36 → 0.41 | 0.21 → 0.27 |
| Phi-3.5 `policy_guided_verifiability` Qwen3-Goedel (§8.1) | 456 | 74 | 33 | 0.362 → 0.434 | 0.54 → 0.59 | 0.41 → 0.49 | 0.26 → 0.33 |
| §10 rerank cache, 150 trajectories (legacy labels) | 1,586 | 84 | 59 | 0.510 → 0.547 | 0.45 → 0.46 | 0.60 → 0.69 | 0.48 → 0.48 |

What changes and what does not: every rate rises by 4–7 points, level 2 most (where
steps carry variables and hypotheses). The §4 level-1 edge of `policy_guided_verifiability`
over `base_greedy` on Phi (0.63 vs 0.54) is untouched; its level-3 deficit narrows
(0.24/0.30 → 0.31/0.33). §7's "Qwen3-1.7B guidance is *worse* than MiniLM by the real
measure" survives on Qwen2.5-7B (0.405 vs 0.434) and on Phi (0.379 vs the MiniLM arm's
0.425). Every real-formalization number in §4, §7, §8 and §10 above should be read with
this +4–7 point correction; the tables there are left as recorded, with this section as
the erratum. (The §9.2 rendering problem is *not* corrected by this: these caches still
formalize the problem for a third of their steps. The step-level re-labelled numbers are
in `forked-tree-classifier-retraining.md` §4.1 and §8.)

## 10. `full_trajectory_rerank` — does natural generation actually fix the mechanism?

§9's confound is one independent problem with `rerank`. A second, raised directly: the
`<step>`-token generation protocol itself (`base_model.py`'s `DEFAULT_SYSTEM_PROMPT`,
"write ONE step, then the marker `<step>`, then stop") is a generation-time constraint
FaithformBench's own training steps never had — those came from a natural,
unconstrained chain-of-thought, segmented *after* generation (the ProcessBench
convention). `full_trajectory_rerank` (`rllf.generation.guided_decoder`) tests this
directly: sample N=10 complete, naturally-written solutions per problem (no `<step>`
forcing), retroactively segment each with the same model, score every candidate's mean
per-step formalizability under a classifier, and compare the classifier's pick against
the population and an oracle — all checked against real Lean, not self-reported.

**Scope**: 15 of the same 110 Qwen2.5-7B-Instruct MATH-500 problems used in §8.3 (5 each
from levels 1/2/3), N=10 trajectories each — 1,586 real-Lean checks total (natural CoT
segments into ~10.6 steps/trajectory vs. `<step>`-mode's ~4, so this pilot cost more
real-Lean checks than the full 110-problem, 3-arm comparison in §8.3 despite covering
7x fewer problems).

| | real-Lean formalization rate |
| --- | --- |
| Population mean (all 150 trajectories) | **53.5%** |
| Oracle best-of-10 | 68.0% |
| Oracle worst-of-10 | 33.7% |
| MiniLM's pick (mean-step-score argmax) | **49.1%** |
| Qwen3-1.7B's pick | **50.8%** |

**The aggregate result confirms the mechanism hypothesis cleanly.** Natural,
unconstrained generation formalizes at 53.5% on average — a **+16.9 point** absolute
jump over `<step>`-token `base_greedy`'s 36.6% on the identical base model and problem
pool (§8.3). The generation-protocol mismatch is real and large: forcing one-step-then-stop
generation costs a substantial amount of real formalizability, independent of any
classifier.

**But neither classifier can find the better trajectories within a natural pool — both
pick *worse* than random.** MiniLM's pick (49.1%) and Qwen3-1.7B's (50.8%) both sit
*below* the population mean (53.5%), not above it. The N-ablation (200 random subsets
per k, reusing the one cached pool) makes this worse, not better, as more candidates
become available:

| k | MiniLM pick | Qwen3-1.7B pick |
| --- | --- | --- |
| 1 | 53.3% | 53.3% |
| 2 | 52.4% | 51.4% |
| 3 | 52.5% | 50.8% |
| 5 | 52.0% | 50.3% |
| 10 | **49.1%** | 50.8% |

At k=1 a "pick" is just the population mean by construction (nothing to select among),
so both classifiers start there as expected. As N grows, MiniLM's selection quality
*degrades monotonically*; Qwen3-1.7B's is roughly flat but never above the k=1 baseline
either. More candidates to choose from makes the reward signal's selection *worse*, the
opposite of what a real, useful ranking signal should do.

**Reading this against §8.3's `<step>`-token result on the same base model and
problems**: `<step>`-mode classifiers were flat-to-mildly-negative relative to
`base_greedy` (MiniLM 36.6% vs. 36.6%; Qwen3-1.7B 35.3% vs. 36.6%) — no better than
baseline, but not clearly worse than *chance* either. Here, both classifiers are clearly
worse than chance (random selection == the population mean). This is a more clearly
negative result for the classifiers specifically, layered on top of a more clearly
positive result for natural generation specifically.

### 10.1 Why the classifiers pick worse than random — measured, not guessed

All of the following is computed on the cached 150 trajectories / 1,586 real-Lean
verdicts (CPU only; no new generation or Lean). Three candidate explanations were
tested; two were refuted.

**Refuted: trajectory length.** (This document's own first guess.) Length has no
relationship with true rate (ρ = −0.05 pooled, +0.04 within-problem), and the
classifiers prefer *shorter* trajectories, not longer (within-problem ρ with step count:
MiniLM −0.20, Qwen3-1.7B −0.29; picks average 8.5–8.9 steps vs. 10.6 population).

**Refuted: rounding.** Natural CoT rounds ("≈ 0.206"), which would make an equality
literally false in Lean — but equation steps containing decimals/≈ prove at 70.7%,
integer-only ones at 43.4%. Whatever makes equation steps fail, it isn't rounding.

**Partly relevant, not the driver: correctness-blindness.** `PROVED` requires the claim to
be *true*, and the verifiability classifier was trained only on unperturbed (all-correct)
FaithformBench steps — it has never seen a wrong-but-well-formed step. Correctness does
dominate provability (steps in correct trajectories prove at 58.2%, in wrong ones 40.3%;
trajectory true rate 0.605 vs 0.379). But the classifiers' picks are slightly *more*
likely correct than random (73% vs. 69%), so this cannot be what pushes them below
random. And for Qwen3-1.7B the within-problem anti-correlation persists restricted to
correct-only trajectories (ρ = −0.16) — it anti-selects on genuine provability among
true solutions, not on correctness.

**The actual structure — a Simpson's-paradox pattern.** Pooled across all 150
trajectories, classifier score correlates *positively* with true rate (MiniLM +0.25,
Qwen3-1.7B +0.21). Within a problem — where selection actually happens, among 10
alternative solutions to the *same* question — it correlates *negatively* (−0.15,
−0.18). The pooled correlation is problem difficulty: easy problems get both higher
scores and higher true rates. The classifier learned between-problem signal, which is
all FaithformBench could teach it: **FaithformBench contributes exactly one trajectory
per problem**, so the classifier never once had to rank two competing solutions of the
same problem, and its held-out AUC never rewarded that. `rerank`,
`full_trajectory_rerank`, and `lean_constrained_beam_search` all ask for exactly the
within-problem comparison it was never trained on.

**Why negative rather than merely zero — the surface proxy.** By step type
(regex-classified), real-Lean proved rate on these natural Qwen2.5-7B steps is:

| step type | n | proved |
| --- | --- | --- |
| setup/declarative, no `=` ("We need to find…") | 131 | **67.2%** |
| boxed final answer | 93 | 62.4% |
| prose, no `=` | 351 | 56.7% |
| multi-equation | 384 | 48.4% |
| single equation | 627 | **44.3%** |

Explicit computational steps are the *least* provable type in this pipeline; declarative
and prose steps the most (and not because setup steps formalize to trivialities — they
prove at 73% inside correct trajectories vs. 47% inside wrong ones, i.e. their
provability tracks the reasoning like any other step). Meanwhile the classifiers'
within-problem preference is almost entirely a preference for **equation density**:
ρ(score, fraction of equation steps) = +0.26 (MiniLM), **+0.49** (Qwen3-1.7B), with the
mirror-image dispreference for setup/prose steps (−0.16/−0.27, −0.19/−0.33). And
equation density has zero-to-negative within-problem correlation with true rate (−0.08),
while setup-step density is *positively* correlated (+0.21). So: the classifier learned
"states its arithmetic explicitly ⇒ formalizable" — a proxy that was genuinely
informative on FaithformBench's step distribution (and is literally what this codebase's
own `DEFAULT_SYSTEM_PROMPT` asked the base model for) — and on natural CoT checked by
Goedel + DeepSeek-Prover-V2, that proxy points the wrong way. The stronger classifier
(Qwen3-1.7B) learned the proxy more sharply (+0.49 vs +0.26) and so applies it more
consistently. Picking "more formalizable-looking" steps selects for the step type this
prover proves least.

**Limitation.** *Why* explicit equation steps fail here — prover search failure on
concrete arithmetic, formalizer mistranslation, or the claim being genuinely false —
cannot be decomposed from this run: the per-step Lean statements/proofs were not stored
(only `proved`/`stage`/`error`, and `error` is empty for prove-stage failures). Only 84 of
777 failures were formalization-stage; the other 693 formalized fine and then no
candidate proof compiled. Re-running with statements kept is the way to settle it.

### 10.2 Is it autoformalizer mismatch? No — and step-level transfer is near chance for every checkpoint

The real-Lean pipeline formalizes with Goedel-Formalizer-V2-8B and proves with
DeepSeek-Prover-V2-7B; the classifiers in §10 were trained on `all8_majority` labels.
FaithformBench's Goedel-only labels come from essentially this pipeline, so the
`prm_verifiability_goedel` checkpoints (offline held-out AUC: MiniLM 0.787, Qwen3-1.7B
0.825) are the clean test. All 1,586 cached steps were scored under ten classifiers
(`full_trajectory_score_all.py`, 0.04 GPU-h — forward passes only, no generation, no Lean)
and compared against the stored real-Lean verdicts (`per_step_scores.json`).

| classifier | step AUC vs real Lean | AUC \| correct trajs | AUC \| formalized | top-1 pick | within-problem ρ | pick − random [95% CI] |
|---|---|---|---|---|---|---|
| minilm_all8 | 0.551 | 0.560 | 0.558 | 0.491 | −0.152 | −0.043 [−0.142, +0.039] |
| **minilm_goedel** | 0.528 | 0.540 | 0.527 | 0.584 | −0.071 | +0.049 [−0.002, +0.101] |
| minilm_af_majority | 0.528 | 0.534 | 0.542 | 0.483 | −0.141 | −0.052 [−0.143, +0.011] |
| minilm_kimina | 0.539 | 0.578 | 0.537 | 0.513 | +0.011 | −0.022 [−0.084, +0.037] |
| minilm_herald | 0.545 | 0.570 | 0.556 | 0.551 | +0.024 | +0.016 [−0.024, +0.063] |
| minilm_stepfun | 0.509 | 0.526 | 0.516 | 0.529 | −0.071 | −0.006 [−0.055, +0.048] |
| minilm_gpt-5.2 | **0.614** | 0.605 | **0.609** | 0.548 | −0.051 | +0.012 [−0.039, +0.069] |
| qwen3_all8 | 0.568 | 0.564 | 0.588 | 0.508 | −0.175 | −0.028 [−0.133, +0.053] |
| **qwen3_goedel** | 0.497 | 0.519 | 0.502 | 0.465 | −0.151 | −0.070 [−0.191, +0.035] |
| minilm_agreement_all8 | 0.608 | **0.628** | 0.579 | 0.576 | −0.113 | +0.040 [−0.013, +0.100] |

Population (random-pick) mean true rate 0.535. Step AUC is over all 1,586 steps; "AUC | correct"
restricts to steps of trajectories with the right final answer (true claims ⇒ measures
provability, not correctness); "AUC | formalized" drops the 84 formalize-stage failures.
Pick and ρ are averaged over the 15 problems; the CI is a 5,000-sample bootstrap over
problems (`full_trajectory_pick_bootstrap.py`).

**Mismatch is rejected.** If the §10 picks were bad because the classifiers were fitted to
an 8-labeler ensemble rather than to this prover, the Goedel-matched checkpoints should
(a) discriminate these steps *better* against the Goedel/DeepSeek verdict and (b) pick
better on both architectures. Neither happens: the Goedel-only classifiers have *lower*
step-level AUC than their all8 counterparts (MiniLM 0.528 vs 0.551; Qwen3-1.7B 0.497 —
exactly chance — vs 0.568), and their within-problem ρ stays negative (−0.071, −0.151).
The pick statistic moves in opposite directions on the two architectures (MiniLM-Goedel
+0.049, Qwen3-Goedel −0.070; paired Goedel−all8 differences +0.093 [+0.002, +0.198] and
−0.043 [−0.143, +0.022]) — with 15 problems that is noise around zero, not a mismatch
effect that a matched labeler repairs. An autoformalizer-matched training set would not
fix §10.

**The larger finding is the first column.** Every checkpoint's step-level discrimination
against real Lean on these natural Qwen2.5-7B-Instruct steps is 0.50–0.61 — against
0.79–0.83 on FaithformBench's own held-out split. The offline AUC does not transfer to
naturally generated CoT under the real pipeline, for any labeler choice, either
architecture, and regardless of whether the trajectory is correct or the step formalized
(the three AUC columns are within ~0.02 of each other for every row, so the classifiers
are not selectively catching formalize-stage failures or wrong-trajectory steps either).
This is the same distribution-shift story as §10.1 seen at step level: what separates
formalizable from non-formalizable steps in FaithformBench (equation density, explicit
arithmetic) is not what separates them on this generator + this prover.

Two weak positives worth noting, neither significant at n = 15: the two checkpoints
*least* tied to prover-run labels transfer best — `minilm_gpt-5.2` (0.614) and the
correctness-trained `minilm_agreement_all8` (0.608 overall, 0.628 on correct
trajectories, the best "provability given the claim is true" number in the table, and a
positive pick gain +0.040 [−0.013, +0.100]). If any of these are pursued they need the
larger validation set (§10.1's recommendation) before a conclusion.

**Caveat on §10's "worse than random."** The bootstrap CIs above put the all8 pick gains at
−0.043 [−0.142, +0.039] (MiniLM) and −0.028 [−0.133, +0.053] (Qwen3-1.7B): the point
estimates are below random but the 15-problem sample cannot distinguish them from zero.
The defensible statement is *no better than random at trajectory selection, with
step-level AUC near chance against real Lean* — the mechanism in §10.1 (Simpson's paradox,
equation-density proxy) describes what the classifiers prefer, not a proven magnitude of
harm. Rerunning §10 at 110 problems (~160 GPU-h at n = 10, ~65 at n = 4) is what would
settle the sign.

---

## 11. Open items / not yet run

- *Done (Oct 16, `forked-tree-classifier-retraining.md` §9): 7 of 300 sampled failures (2.3%)
  are refutable by the same prover, 0/90 controls — "not proved" is search/statement
  failure, not demonstrated falsity.* **Formalizability vs. correctness**: quantify how much of "not proved" is genuine
  falsity vs. search failure — needed to know what the Real-Formalizability metric is
  actually measuring (§4).
- **What's specifically driving Qwen3-1.7B's easy-problem transfer behavior** — §8.2 showed
  it isn't fixed by dropping the label ensemble, and §8.3 showed it doesn't just improve with
  a more in-distribution base model, it *reverses sign* (+5.8 on Phi-3.5-mini → −6.8 on
  Qwen2.5-7B-Instruct). Neither hypothesis tested so far explains this; still open.
- **Finish `policy_guided_agreement`** at 110-problem scale (30% done, paused at a
  16-hour job time limit), and re-run the level-stratified breakdown once it lands.
- *Larger encoder on the forked trees — done (Oct 15, `forked-tree-classifier-retraining.md`
  §8):* Qwen3-1.7B lifts the forked classifier's real-Lean step AUC to 0.72 on the
  generator's MATH-500 steps and 0.65 on the natural-CoT cache (MiniLM 0.63 / 0.57, old
  verifiability 0.64 / 0.60) — capacity was the judge's limit — but constrained beam search
  with it is null on accuracy (0.627 vs 0.655 search-only) and +2–3 points formalization
  (only the proved-step count significant), and its rerank pick is more formalizable yet
  less often correct. Judge quality is no longer the bottleneck; the intervention strength
  (α / `tau_phi`) and the easy problem set are.
- **Revisit the agreement setting (mode B) with a larger encoder.** Parked while the
  forked-tree retraining runs, but worth returning to: (a) offline, agreement AUC rose
  with encoder size where verifiability did not — MiniLM-L6 0.544 → BERT-base 0.617 (§1),
  vs. 0.798 → 0.798 for MiniLM → ModernBERT on verifiability — so the near-chance
  agreement result reads as a *capacity* limit, and the fine-tuned Qwen3-1.7B encoder
  that lifted verifiability from 0.798 to 0.863 (§7) has never been fitted in agreement
  mode; (b) agreement is the mode whose labels survive the faithfulness problem in the
  benchmark's own verdicts (19% of false-by-construction steps are "Proved" — it keeps a
  step only when the verdict matches the known ground truth, see the FaithformBench column
  audit); (c) it was the checkpoint that transferred best to real Lean in §10.2 (step AUC
  0.608, 0.628 on correct trajectories, the only positive pick gain of the MiniLM family),
  albeit against the now-known problem-level labels. Plan, in order:
  1. `scripts/run_faithform.py --stage fit --mode agreement --encoder Qwen/Qwen3-1.7B
     --pooling last --settings all8_majority` (same recipe as the §7 verifiability fit,
     same fit/held-out split, seed 0) → held-out agreement AUC; also `af_majority` and
     `llm_majority`, since the LLM verdicts have the lower false-"Proved" rate (9–12% vs
     19–23%) and may give cleaner agreement labels.
  2. If AUC clears ~0.70: score the §10 cached 1,586 steps and the forked-tree trees with
     it (forward passes only, ~0.05 GPU-h) against the *re-labelled* step-level real-Lean
     verdicts, alongside the forked-tree classifiers, using
     `rllf.evaluation.forked_tree_eval` — this is the apples-to-apples transfer test.
  3. Only then a decoding run: `policy_guided_agreement` uncapped on the 110-problem
     subset, both base models, plus the finished 110-problem real-Lean comparison above.
  Cost: step 1 ≈ 1–2 GPU-h per setting, step 2 negligible, step 3 ≈ 10 GPU-h decode +
  ~15 GPU-h Lean.
- **`rejection` mode** — Lean-in-the-loop resampling during decoding (verify top
  candidates, resample on failure) instead of rerank-and-commit. Not yet tried on either
  base model; a natural next step given §2's (and now §7's) null result under plain rerank,
  now doubly motivated since §7 rules out classifier quality as the explanation.
- **`token`-level guidance** — per-token `log p_base + α·log π_φ` steering. Not yet tried.
- **`full_trajectory_rerank` at scale** (§10) — the length-confound and
  autoformalizer-mismatch hypotheses are both refuted (§10.1, §10.2); what remains is the
  15-problem sample size: pick CIs are ±0.05–0.10, so the sign of the selection effect
  is not settled. Re-run on the 110-problem set with per-step Lean statements stored
  (~65 GPU-h at n = 4), which also unblocks the failure-decomposition in §10.1's
  limitation. Any within-problem retraining (§10.1) needs this set as its validation gate.
  *Update (Oct 14):* the 15-problem cache is re-labelled under the step-level `facts`
  rendering (`forked-tree-classifier-retraining.md` §4.1): per-step agreement with the
  §10 labels is only 66.6% (proved 0.510 → 0.576; 52 of the 84 formalize failures now
  prove), every classifier's pick − random is within ±0.05 with CIs through zero, and the
  forked-tree classifier's step AUC on these natural steps is 0.572 (old verifiability
  0.599). The §10 / §10.2 tables above are legacy-rendering numbers; re-read them with that
  agreement figure in mind. The 110-problem run remains the open validation gate.
- *Done (Oct 16, §9.4):* **Re-run the formalize-stage failures of every real-Lean cache with the fixed
  `_THEOREM_RE`** (`forked-tree-classifier-retraining.md` §7.2): the old regex rejected
  every Goedel statement with binders, so 84 (§10) + 79/74 (§8.1/§8.2) + 154 (§4) + 216
  (§8.3) "formalize" failures were never sent to the prover — ~607 steps, ~7 GPU-h. Until
  then every real-formalization rate in this file is a lower bound biased against
  symbolic/contextual steps (the §10.1 step-type breakdown in particular).
- **Retraining on the decoded model's own competing solutions** — in progress, see
  `forked-tree-classifier-retraining.md`: step-native Qwen2.5-7B-Instruct, 150 MATH-train
  problems × 4 forked trajectories, every unique step real-Lean labelled (statements
  stored), then within-problem pairwise/listwise/ORM fits (experiment 1) and a
  boundary-trained classifier for `lean_constrained_beam_search` (experiment 2).
- **Re-run §2/§3/§4's accuracy and formalizability numbers uncapped** at full scale — §9's
  re-run covered only `base_greedy`/`self_consistency`/`policy_guided_verifiability` on
  the two base models; `random_of_k`, `policy_guided_agreement`, the weighted
  self-consistency variants, and the real-Lean/self-reported formalizability metrics in
  §3/§4/§8 all still reflect the capped measurement.
