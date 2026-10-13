# RLLF — Interim Report

A Lean-supervised process reward model for mathematical reasoning: can a reward model
trained on real-autoformalizer verdicts improve how a base LLM decodes math solutions?
This report covers what data went into fitting and testing it, every result obtained so
far, and what's still open. Two companion documents carry the full detail this report
summarizes: `settings_comparison.md` (the offline FaithformBench result) and
`experiments_v1.md` (every decoding-time experiment, with exact numbers and decoder
configuration).

---

## 1. Data

### Fitting the reward models — FaithformBench

- **27,816 step-variant labels over 1,179 problems**, from real autoformalizers +
  DeepSeek-Prover-v2 verdicts — no rule-based formalizer anywhere in the training labels
  (a rule-based formalizer exists in the codebase but is retained for tests/CI only; an
  earlier experiment showed it inflates AUC via surface pattern-matching, not real
  formalizability).
- **8 label sources**: 4 dedicated autoformalizers (Goedel, Kimina, Herald, StepFun) + 4
  general-purpose LLM judges (GPT-5.2, Claude-Opus-4-7, Gemini-3.1-Pro, Qwen-Plus), each
  usable alone or majority-voted (`af_majority`, `llm_majority`, `all8_majority`).
- **Two label settings** built from the same source data: **mode A (verifiability)** —
  "can this step be formalized and proved" (unperturbed steps only); **mode B
  (agreement)** — "is this step actually correct" (true step PROVED vs. corrupted step
  REFUTED, everything else discarded).
- **80% / 20% held-out split by problem** (no problem's steps appear on both sides).
- **46 MATH-500 problems overlap with FaithformBench** — excluded from the fitting pool,
  kept whole in the MATH-500 decoding benchmark (a leakage trap documented in the
  codebase's own handoff notes).
- **Encoders**: MiniLM-L6 (default), BERT-base-uncased and ModernBERT-base (capacity
  controls).

### Testing decoding — MATH-500

- **HuggingFaceH4/MATH-500**, a 250-problem subsample stratified by difficulty level,
  fixed seed, identical across every decoding experiment so results are paired and
  comparable.
- **Two base (frozen, undecoded) models**, run fully independently end to end:
  `microsoft/Phi-3.5-mini-instruct` (3.8B) and `Qwen/Qwen2.5-7B-Instruct` (7B).
- **Decoder**: `rerank` mode only so far (sample 4 candidate steps, score with the
  reward model, keep the top one) — see §3 for why this matters and what's not yet tried.

### Real-Formalizability ground truth (new this week)

- **`Goedel-LM/Goedel-Formalizer-V2-8B`** writes a Lean statement for a step;
  **`deepseek-ai/DeepSeek-Prover-V2-7B`** attempts a tactic proof (n=16 samples/step); the
  model's tactic block is spliced onto the *original, untouched* statement and compiled
  by a real `lake exe repl` against a fully built **Mathlib** — genuine ground truth, the
  same methodology FaithformBench itself used, run fresh on our own generated steps
  rather than on FaithformBench's fixed set.
- Currently covers `base_greedy`, `policy_guided_verifiability`, and
  `policy_guided_agreement` on Phi-3.5-mini, on a matched MATH-500 subsample (same
  questions across all three arms); being scaled from 20 → 110 problems now (§2.4).

---

## 2. Results obtained so far

### 2.1 Which label setting is actually learnable? (offline AUC)

| encoder | mode A (verifiability) | mode B (agreement) |
| --- | --- | --- |
| MiniLM-L6 (33M) | **0.798** | 0.544 |
| BERT-base (110M) | — | 0.617 |
| ModernBERT-base (150M) | 0.798 (identical) | — |

Mode A is real, learnable, and **capacity-saturated** — a 2024, 4.5x-larger encoder gets
the identical score. Mode B is near-chance with a small, capacity-recoverable component
(+0.073 from more capacity, but nowhere near mode A). Ensembling (`all8_majority`)
**helps** mode A (best setting, 0.798) and **hurts** mode B (worst setting, 0.508) — the
two modes measure genuinely different things, not the same signal at different noise
levels. Full formalizer-by-formalizer table and the "kind vs. degree" argument:
`settings_comparison.md`.

### 2.2 Does either policy improve MATH-500 answer accuracy?

> **⚠ Superseded by a corrected, uncapped re-run — see `experiments_v1.md` §9.** The
> numbers below were measured under a `max_steps=6` cap that silently gave 19–52% of
> every arm's trajectories an empty (guaranteed-wrong) final answer, at a rate that
> differed systematically by arm. The re-run confirms the confound was real: every
> arm's true accuracy is substantially higher than shown below, and `self_consistency`'s
> statistical edge over `base_greedy` — the most confidently-stated finding in this whole
> report — drops to borderline (Phi-3.5-mini, p=0.049) or disappears entirely
> (Qwen2.5-7B, p=0.121) once corrected. The one thing that **does** survive unchanged:
> `policy_guided_verifiability` stays statistically indistinguishable from `base_greedy`
> on both models, both before and after correction.
>
> **Second correction (`experiments_v1.md` §9.1):** the answer grader rejected correct
> LaTeX answers (`\(\frac{6}{5}\)` vs `\frac65`, `76` vs `76^\circ`). Re-graded uncapped
> accuracies: Phi-3.5-mini 0.376 / 0.432 (p=0.059) / 0.364 (p=0.780); Qwen2.5-7B
> 0.664 / 0.696 (p=0.243) / 0.640 (p=0.451) for `base_greedy` / `self_consistency` /
> `policy_guided_verifiability`. Ordering and conclusions unchanged; `self_consistency`'s
> edge is now non-significant on both models.

**No — on either base model, at any base-model strength tested** (under the original,
since-corrected measurement; see the box above and `experiments_v1.md` §9 for what
actually holds up).

| arm | Phi-3.5-mini | Qwen2.5-7B |
| --- | --- | --- |
| `base_greedy` | 0.256 | 0.360 |
| `random_of_k` | 0.260 (p=1.000) | 0.320 (p=0.143) |
| `self_consistency` | **0.344 (p<0.001)** | **0.440 (p=0.002)** |
| `policy_guided_verifiability` | 0.268 (p=0.736) | 0.356 (p=1.000) |
| `policy_guided_agreement` | 0.244 (p=0.736) | 0.336 (p=0.392) |

Only `self_consistency` — majority vote over whole solutions, no reward model involved —
beats greedy decoding, on both models. Doubling the base model's raw accuracy
(Phi-3.5-mini → Qwen2.5-7B) left the reward-guided arms exactly as flat relative to their
own baseline, ruling out "the base model has no headroom" as the explanation. Weighting
`self_consistency`'s vote by either policy's reward (instead of counting each trajectory
as 1) never helps, and measurably hurts on Phi-3.5-mini where the reward signal is real.
Full tables and McNemar tests: `experiments_v1.md` §2.

### 2.3 Predicted formalizability — the "free win," self-graded

Even with flat accuracy, does `policy_guided_verifiability` produce steps the policy
itself calls more formalizable? **Yes, clearly, on both models:**

| arm | Phi-3.5-mini fully-formalizable rate | Qwen2.5-7B fully-formalizable rate |
| --- | --- | --- |
| `base_greedy` | 8.0% | 8.8% |
| **`policy_guided_verifiability`** | **20.4%** | **21.6%** |
| `policy_guided_agreement` | 5.6% | 8.8% |

More than double the fully-formalizable-trajectory rate, replicated on both base models,
at statistically unchanged answer accuracy. (Mode B's version of this metric is
degenerate — 100% everywhere — a direct consequence of its near-chance AUC, not a
finding about the decoding arms.) Full tables: `experiments_v1.md` §3.

### 2.4 Real-Formalizability — the same claim, checked against real Lean, not the policy's opinion

> **⚠ Superseded — the real-Lean pipeline was formalizing the *problem*, not the step
> (`experiments_v1.md` §9.2).** Every real-Lean number in §2.4–§2.7 was produced by
> handing Goedel `Problem: … / Previous steps: … / Step to verify: …` under a prompt that
> asks for "the following problem statement"; for most steps it returned the problem or
> its answer ("f(−2) = 20−6+4" → `f(−2) = 18`; "Definition: cot θ = cos θ/sin θ" →
> `cot(−π/3) = −√3/3`). In the caches that kept statements, 33–38% of steps share a
> verbatim Lean conclusion with another step of the same problem (a lower bound). The
> "true formalization rate" of a trajectory was therefore largely per-problem provability
> plus prover sampling noise — which also explains §2.7's within-problem null more simply
> than the equation-density proxy. Unaffected: the offline AUCs (§2.1, §2.5) and all
> accuracy numbers (§2.2). A step-level claim rendering has been found
> (`forked-tree-classifier-retraining.md` §7.2) and the affected caches will be
> re-labelled under it; until then treat §2.4–§2.7 as method description, not results.

The result that matters: is §2.3's gap real, or is the policy just grading its own kind
of output favorably? **The honest answer needed two corrections to get right, both
caught by direct questions rather than by careful-enough analysis the first time** —
worth recording plainly since the final answer is more interesting than either wrong
intermediate one.

First pass (20 problems): looked like a clean confirmation, +10.9-point real edge for
`policy_guided_verifiability`. Second pass (aggregated to 110 problems): looked like the
edge had completely vanished (37.2% vs. 36.5%). **Both were misleading.** MATH-500's
stratified sample is built difficulty-level-by-level in contiguous blocks, not shuffled —
the 20-problem pilot was *entirely* level-1 (easiest) problems, while the 90 problems
added to reach 110 were 98% level-2/3. The 110-problem *aggregate* mixed two different
difficulty populations, and harder problems (dominating the total) diluted the real
easy-problem signal to invisibility. Splitting by level instead of aggregating:

| level | `base_greedy` | `policy_guided_verifiability` | Δ |
| --- | --- | --- | --- |
| 1 (easy) | 54.2% | **63.2%** | **+9.0 pts** |
| 2 (medium) | 37.5% | 38.9% | +1.4 pts (noise) |
| 3 (hard) | 29.4% | 24.7% | -4.7 pts |

The level-1 result (63.2%/54.2%) closely replicates the original pilot (64.1%/53.2%) —
it's the same real effect, seen twice, not noise. **The corrected finding**:
`policy_guided_verifiability`'s real formalizability edge is genuine and consistent on
easy problems, and disappears (or mildly reverses) as difficulty increases. Neither
"confirmed" nor "refuted" was right; "real but difficulty-dependent" is.

**A second, still-open question**: does "proved" in this pipeline measure formalizability
or something closer to correctness? FaithformBench's mode-A only ever tested already-true
steps, so it never had to separate the two; here, freshly-generated steps can simply be
wrong, and a wrong claim should fail to prove regardless of its formalizability. Failure
logs show both genuine falsity and pure search failures among "not proved" cases, but not
rigorously quantified yet — an open item, not a resolved one.

Five real bugs in the verification pipeline itself (byte-level-BPE decode corruption, a
`let`-binding parsing bug, a fundamentally-wrong "compare the model's restated theorem"
design replaced by splicing the model's proof onto the untouched original statement, and
a double-header Lean syntax error) were found and fixed getting this pipeline to
genuinely work — documented in `experiments_v1.md` §4 in case anyone else builds on the
same reused infrastructure ([VeriForm](https://github.com/Ighina/VeriForm)).

### 2.5 Does a much stronger verifiability classifier change anything? No.

Two follow-up questions: was MiniLM/ModernBERT's identical 0.798 AUC (§2.1) a ceiling from
an easy-skewed training set, and would a genuinely bigger, differently-architected
classifier do better? Checked both. The training pool is not easy-skewed (65.5% of
MATH-derived steps are level 3–5; ~72% of the full pool is hard/olympiad material). A
fine-tuned **Qwen3-1.7B** classifier (causal-decoder backbone, last-token pooling) does
break the ceiling: **AUC 0.863**, and — unlike MiniLM, which degrades from 0.810 to 0.722
across MATH levels 1→5 — its advantage holds flat to *widening* with difficulty (+0.111 AUC
on the hardest MATH tier).

Plugged into the same `policy_guided_verifiability` MATH-500 run (Phi-3.5-mini, identical
250-problem sample): **accuracy is unchanged** (0.272 vs. the old policy's 0.268, both
statistically indistinguishable from `base_greedy`'s 0.256, p=0.597). The self-reported
formalizability "free win" replicates almost exactly (~2.5x lift in fully-formalizable rate
under either classifier) — **but checked against real Lean, it reverses**: Qwen3-1.7B-guided
decoding real-formalizes *worse* than the original MiniLM-guided run at every difficulty
level (level 1: +5.8 vs. +9.0 pts over `base_greedy`; level 3: −8.4 vs. −4.7 pts), and its
aggregate rate (33.0%) sits below `base_greedy`'s (37.2%) — unlike MiniLM's, which roughly
matched it (36.5%). A classifier with uniformly better, more difficulty-robust AUC produced
a worse real downstream result. Calibration was the first suspect and is ruled out by the
code itself: `rerank`'s greedy candidate choice is `argmax` over one policy's own scores, and
any monotonic recalibration (Platt scaling included) leaves an argmax over a fixed set
unchanged — so the two classifiers must genuinely *disagree* on which steps are more
verifiable, and that disagreement generalizes worse for Qwen3-1.7B despite its better
held-out AUC. **Classifier capacity was never the bottleneck for the null accuracy result,
and better AUC did not transfer to better real formalizability either** — full detail,
including two CPU/GPU-device bugs found fixing this, in `experiments_v1.md` §7.

### 2.6 Three follow-up investigations into the generalization gap

Chasing why Qwen3-1.7B's real-Lean transfer is worse than MiniLM's despite better AUC:

- **Candidate-level disagreement** (§8.1): the two classifiers agree on only 44.9% of
  identical candidate pools — not explained by candidate length, U-shaped by difficulty
  (worst at the extremes), and manual inspection of disagreements found no consistent
  directional bias (neither classifier is systematically favoring trivial or wrong steps).
- **Ensemble labels vs. single source** (§8.2): re-fitting Qwen3-1.7B on one label source
  instead of the 8-way majority vote pulls its hard-problem loss and aggregate transfer
  almost exactly onto MiniLM's numbers — real, partial support for ensemble-overfitting. But
  its easy-problem edge doesn't recover at all (drops to +0.0, even below the ensemble
  version's +5.8) — something else, likely intrinsic to Qwen3-1.7B, explains that part.
- **Base model identity** (§8.3): Qwen2.5-7B-Instruct is confirmed as one of FaithformBench's
  own 8 generator models (351/1179 problems), unlike Phi-3.5-mini. Re-running the full
  3-arm real-Lean comparison there **confirms the distribution-shift hypothesis on the
  metrics that matter most** — MiniLM's aggregate transfer becomes essentially perfect
  (+0.0), Qwen3-1.7B's improves from −4.2 to −1.3, and both classifiers' hard-problem loss
  shrinks substantially. **But Qwen3-1.7B's easy-problem edge doesn't just shrink, it flips
  sign** (+5.8 on Phi-3.5-mini → −6.8 on Qwen2.5-7B-Instruct) — a genuinely new, unexplained
  result on top of an otherwise-confirmed hypothesis.

Full tables and the side-by-side comparisons: `experiments_v1.md` §8.

### 2.7 Does natural (non-`<step>`-forced) generation fix the mechanism? Partly.

`full_trajectory_rerank`: sample N=10 complete, naturally-written solutions per problem
(no `<step>`-token forcing), retroactively segment with the same model, score each by
mean per-step formalizability, compare the classifier's pick against the population and
an oracle — checked against real Lean. 15 problems, 1,586 real-Lean checks.

**Confirms the mechanism hypothesis on the aggregate**: natural generation formalizes at
53.5% on average, a **+16.9 point** jump over `<step>`-token `base_greedy`'s 36.6% on the
same base model and problems. The generation-protocol mismatch (§2.5's original
diagnosis, and `lean-constrained-beam-search.md` §1) is real and costs a lot on its own.

**But neither classifier can pick the better trajectories from a natural pool — both do
worse than random** (MiniLM 49.1%, Qwen3-1.7B 50.8%, vs. 53.5% population mean), and an
N-ablation shows this *worsening*, not improving, as more candidates become available.
A more clearly negative result for the classifiers than the `<step>`-token setting
produced (there they were flat, not below chance).

**Why, measured on the cached data** (`experiments_v1.md` §10.1): not length (refuted —
no correlation with true rate, and the classifiers prefer *shorter* trajectories), not
rounding (refuted — decimal equation steps prove *more*), and only partly correctness
(the classifiers' picks are slightly more likely correct than random, and Qwen3-1.7B's
anti-correlation survives among correct-only trajectories). The real structure is a
Simpson's paradox: classifier score correlates positively with true rate *across*
problems (+0.21/+0.25 — it learned problem difficulty) but negatively *within* a problem
(−0.15/−0.18 — where selection actually happens). FaithformBench has exactly one
trajectory per problem, so the classifier was never trained to rank competing solutions
to the same question. And within a problem its preference is almost entirely for
**equation-dense** trajectories (ρ +0.26 MiniLM, +0.49 Qwen3-1.7B) — a proxy for
"formalizable" that pointed the right way on FaithformBench but the wrong way here:
explicit equation steps are the *least* provable type in this pipeline (44%), declarative
setup steps the *most* (67%). Choosing "formalizable-looking" steps selects the step
type this prover proves least.

**Not autoformalizer mismatch** (`experiments_v1.md` §10.2): scoring all 1,586 steps
under ten classifiers, the Goedel-only checkpoints — trained on essentially this
pipeline's labels — discriminate these steps *worse* against the real-Lean verdict than
the all8 ones (step AUC MiniLM 0.528 vs 0.551; Qwen3-1.7B 0.497 vs 0.568), keep a negative
within-problem rank correlation, and their pick gains go in opposite directions on the two
architectures (+0.05 / −0.07, CIs spanning zero). The larger result is that *every*
checkpoint's step-level AUC against real Lean on natural Qwen2.5-7B steps is 0.50–0.61,
versus 0.79–0.83 offline: the FaithformBench discrimination does not transfer to this
generator + prover for any labeler or architecture. Two weak, non-significant positives:
the GPT-5.2-labeled (0.614) and correctness-trained agreement (0.608; 0.628 on correct
trajectories, pick +0.04) checkpoints transfer best. Caveat: with 15 problems the
trajectory-level pick CIs are ±0.05–0.10, so "worse than random" should be read as "no
better than random, with step-level AUC near chance."

---

## 3. What's left

**The uncapped `max_steps` re-run (§2.2) is done** — see `experiments_v1.md` §9 for the
full corrected numbers. Two natural follow-ups it opens rather than closes:
- Re-run the remaining arms (`random_of_k`, `policy_guided_agreement`, the weighted
  self-consistency variants) uncapped too — only the 3 highest-priority arms were
  re-run so far, on cost grounds.
- Re-check §2.3's self-reported formalizability numbers and §2.4/§2.6's real-Lean
  comparisons uncapped — those were all measured on the same capped trajectories and
  inherit the same confound, just not yet quantified for those specific metrics.

**`full_trajectory_rerank` at scale (§2.7)**: length and autoformalizer mismatch are
both ruled out; the open question is now sample size — 15 problems leave the sign of the
selection effect unsettled (CIs ±0.05–0.10). Re-run on the 110-problem set with Lean
statements stored (~65 GPU-h at n = 4); that set is also the validation gate for any
within-problem retraining.

**Finishing and confirming the corrected (difficulty-stratified) Real-Formalizability picture:**
- Quantify formalizability vs. correctness conflation in the "proved" metric (tag each
  failure as genuine falsity vs. search failure) — needed to know what's actually being
  measured before trusting the level-1 effect size at face value.
- Explain Qwen3-1.7B's easy-problem transfer behavior (§2.6) — not fixed by dropping the
  label ensemble, and it *reverses sign* rather than improving on a more in-distribution
  base model (+5.8 pts on Phi-3.5-mini → −6.8 pts on Qwen2.5-7B-Instruct). Neither
  hypothesis tested so far explains this.
- Finish `policy_guided_agreement` at 110-problem scale (30% done, paused at a 16-hour
  job time limit, not a crash — checkpointed and resumable), and redo the level-stratified
  breakdown once it lands, and for §2.3's self-reported numbers too (does the same
  easy-problems-only pattern hide in that aggregate as well?).
- **Come back to the agreement setting with the Qwen3-1.7B encoder.** Offline, agreement
  AUC grew with encoder size (MiniLM 0.544 → BERT-base 0.617) where verifiability did not,
  so its near-chance result looks like a capacity limit, and the fine-tuned Qwen3-1.7B
  encoder (0.863 on verifiability) has never been fitted in that mode. Agreement is also
  the label that survives the benchmark's own faithfulness problem (19% of
  false-by-construction steps are "Proved"; agreement keeps only verdicts that match the
  known ground truth) and the checkpoint that transferred best to real Lean in §2.7.
  Plan: fit (`--mode agreement --encoder Qwen/Qwen3-1.7B --pooling last`, all8 / AF-only /
  LLM-only labels) → transfer test on the re-labelled step-level Lean caches → only then an
  uncapped `policy_guided_agreement` decode. Full plan and costs: `experiments_v1.md` §11.
- Real-Formalizability beyond 110 problems, if the result still seems worth firming up
  further, with level-balanced (not prefix-sliced) sampling this time — full 250 costs
  ~42 GPU-hours at n=16 samples/step, more than a week's shared GPU budget in one shot.
- Real-Formalizability on Qwen2.5-7B (currently Phi-3.5-mini only) — does the same
  easy-problems-only effect pattern hold on a stronger base model?

**Decoder modes not yet tried** (`rllf/generation/guided_decoder.py` implements all
four; only `plain` and `rerank` have been used):
- **`rejection`** — verify top candidates in real Lean during decoding, resample on
  failure, instead of rerank-and-commit. The most natural next step given §2.2's null
  result: `rerank` throws away a wrong step outright with no second chance; `rejection`
  might behave very differently since it can actually reject and retry against ground
  truth mid-generation, not just the policy's prediction.
- **`token`-level guidance** — per-token `log p_base + α·log π_φ` steering, a
  fundamentally different (continuous, not discrete-candidate) guidance mechanism.

**Beyond MATH-500 — diverse datasets:**
- Everything above is one benchmark (MATH-500) and one problem type (competition math).
  An earlier, now-superseded experiment used ProcessBench with a rule-based reward
  (`PROCESSBENCH_EXPERIMENT.md`) — worth deciding whether a modern rerun with the current
  reward models is informative, or whether a different benchmark entirely (GSM8K for an
  easier regime where the base model has more headroom; AIME for a harder one) would
  better stress-test whether §2.2's null result is MATH-500-specific or general.

**Possible response to the null accuracy result itself**, since `rerank` + this reward
signal has now been ruled out twice (two base models):
- A reward model trained on outcome-correctness rather than formalizability (mode B
  didn't work at MiniLM/BERT-base capacity, but hasn't been tried as a target for a much
  larger encoder, or with a differently-sourced correctness label).
- Different K (candidates per step) / step-count budgets than the K=4/6-steps used
  throughout — untested whether the null result is budget-specific.
