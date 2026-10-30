# RLLF — consolidated results and paper outline

Status: Oct 30, 2026 (residual and attention-head probe experiments added, FT §12). One table per question, every number on the *corrected* footing
(uncapped decoding, balanced-brace + sympy answer grading, step-level Lean rendering where
it exists; legacy-rendering numbers are marked and carry the §9.4 binder correction). Detail
and provenance: `experiments_v1.md` (§1–§11) and `forked-tree-classifier-retraining.md`
(§1–§11). GPU budget of the whole programme ≈ 600 GPU-h; this final round (Oct 13–19) ≈ 250.

## A. Does a Lean-supervised step reward improve answer accuracy?

MATH-500, paired per problem. "rerank" = sample 4 candidate steps, keep the reward model's
top one (`policy_guided_verifiability`); `beam` = beam 4 × expansion 2 with the classifier
term (α = 1, mean aggregation); hybrid = beam with real Lean on the judge's unsure steps.

| setting | base model | problems | control | with reward model | Δ (paired) | source |
|---|---|---|---|---|---|---|
| rerank, FaithformBench MiniLM | Phi-3.5-mini | 250 (all levels) | greedy 0.376 | 0.364 | −0.012 | §9 uncapped, re-graded |
| rerank, FaithformBench MiniLM | Qwen2.5-7B | 250 | greedy 0.664 | 0.640 | −0.024 | §9 uncapped, re-graded |
| self-consistency (no reward model) | Phi / Qwen | 250 | greedy 0.376 / 0.664 | **0.432 / 0.696** | +0.056 / +0.032 | §9 — the only method that beats greedy |
| rerank, FaithformBench Qwen3-1.7B (AUC 0.863 offline) | Phi-3.5-mini | 250 (capped run) | greedy 0.256 | 0.272 | +0.016, p = 0.60 | §7 (capped; re-graded direction unchanged) |
| beam, forked MiniLM judge | step-native Qwen2.5-7B | 110 (L1–3) | beam_lm 0.655 | 0.673 | 9/7, p = 0.80 | FT §5.2 |
| beam, FaithformBench MiniLM judge | step-native Qwen2.5-7B | 110 (L1–3) | beam_lm 0.655 | 0.645 | 9/10, p = 1.0 | FT §5.2 |
| beam, forked **Qwen3** judge, α = 1 | step-native Qwen2.5-7B | 110 (L1–3) | beam_lm 0.655 | 0.627 | 8/11, p = 0.65 | FT §8.3 |
| beam, forked Qwen3 judge, α = 4 / 10 / 30 | step-native Qwen2.5-7B | 110 (L1–3) | beam_lm 0.655 | 0.564 / 0.582 / 0.555 | 8/18 p = 0.08; 14/22; 13/24 p = 0.10 | FT §8.6 — stronger steering *hurts* |
| **hybrid**: beam + real Lean on unsure steps (budget 24) | step-native Qwen2.5-7B | 110 (L1–3) | beam_lm 0.655 | **0.655** | 11/11 | FT §10.1 |
| beam, **Goedel residual-stream probe** as judge (α = 1) | step-native Qwen2.5-7B | 110 (L1–3) / 110 (L4–5) | beam_lm 0.655 / 0.382 | 0.600 / 0.364 | 9/15, p = 0.31 / 6/8, p = 0.79 | FT §12.2 |
| beam, **Goedel attention-head probe** (64 benchmark-selected heads) as judge | step-native Qwen2.5-7B | 110 (L1–3) / 110 (L4–5) | beam_lm 0.655 / 0.382 | 0.627 / 0.355 | 7/10, p = 0.63 / 9/12, p = 0.66 | FT §12.4 |
| full-trajectory rerank, forked Qwen3 judge (10 natural CoT trajectories) | Qwen2.5-7B | 15 (L1–3) | random pick 0.753 | 0.667 | −0.087 [−0.23, +0.05] | FT §8.4 |
| full-trajectory rerank, forked Qwen3 judge (4 forked trajectories) | step-native Qwen2.5-7B | 30 held-out MATH-train trees | random 0.275 | 0.267 | ≈ 0 | FT §8.1 |
| **full-trajectory rerank, forked Qwen3 judge** | step-native Qwen2.5-7B | **60 MATH-500 L4–5** | random 0.308 (oracle 0.467) | **0.383** | **+0.075 [+0.013, +0.138]** | FT §11.1 |
| same, forked MiniLM judge / FaithformBench judge | step-native Qwen2.5-7B | 60 L4–5 | random 0.308 | 0.333 / 0.283 | +0.025 [−0.06, +0.11] / −0.025 | FT §11.1 |
| beam, forked Qwen3 judge, α = 1 | step-native Qwen2.5-7B | **110 L4–5** | beam_lm 0.382 (greedy 0.255) | 0.373 | 7/8, p = 1.0 (beam vs greedy 24/10, p = 0.02) | FT §11.2 |

**Answer:** on easy-to-medium problems (MATH-500 levels 1–3), no — not with any judge,
any strength, or real Lean in the loop; beam search itself is worth ~+6 points over
greedy. On hard problems (levels 4–5) the Qwen3 judge trained on the generator's own
Lean-labelled steps gives the first selection gain in accuracy when it reranks *complete*
trajectories (+7.5 points, CI excluding zero, one 60-problem sample); steering the beam
step by step with the same judge does not move accuracy there either (beam search itself
is worth +13 points on hard problems).

## B. Does it make the reasoning more formalizable (real Lean, not self-reported)?

| setting | problems | control | with reward model | paired test | source |
|---|---|---|---|---|---|
| rerank, FaithformBench MiniLM (legacy rendering, binder-corrected) | Phi, 110 | greedy 0.418 (L1 0.54) | 0.425 (L1 **0.63**) | level-1 edge replicated twice | §4, §9.4 |
| rerank, FaithformBench Qwen3-1.7B (legacy, corrected) | Qwen2.5-7B, 110 | greedy 0.433 | 0.405 (MiniLM judge 0.434) | better offline AUC → *worse* real result | §7, §8.3, §9.4 |
| beam, forked MiniLM judge (step rendering) | 110 L1–3 | beam_lm 0.526 step rate / 0.522 word share | 0.552 / 0.534 | p = 0.22 / 0.54 | FT §5.3 |
| beam, forked Qwen3 judge α = 1 | 110 L1–3 | 0.526 / 0.522 | 0.557 / 0.543 | p = 0.08 / 0.22 | FT §8.3 |
| beam, forked Qwen3 judge α = 4, 10 | 110 L1–3 | 0.526 / 0.522 | 0.558 / 0.544; 0.546 / 0.531 | n.s. | FT §8.6 |
| **hybrid (Lean in the loop)** | 110 L1–3 | 0.526 / 0.522 | **0.588 / 0.580** | **p = 0.006 / 0.004**; vs judge-only p = 0.04 | FT §10.1 |
| full-trajectory rerank, Qwen3 judge, natural CoT | 15 L1–3 | random 0.606 | 0.683 | +0.076 [+0.013, +0.138] | FT §8.4 |
| full-trajectory rerank, Qwen3 judge, hard forked trees | 60 L4–5 | random 0.348 | 0.408 | +0.060 [+0.032, +0.090] | FT §11.1 |
| beam, forked Qwen3 judge α = 1, **hard problems** | 110 L4–5 | beam_lm 0.371 / 0.342 | **0.404 / 0.386** | **p = 0.04 / 0.018** | FT §11.2 |
| beam, Goedel residual-stream probe as judge, easy | 110 L1–3 | 0.526 / 0.522 | 0.536 / 0.514 | p = 0.77 / 0.68 | FT §12.2 |
| beam, Goedel residual-stream probe as judge, **hard** | 110 L4–5 | 0.371 / 0.342 | 0.403 / 0.379 | p = 0.12 / 0.083 (= Qwen3 arm's 0.404 / 0.386) | FT §12.2 |
| beam, Goedel attention-head probe as judge, easy / hard | 110 / 110 | 0.526 / 0.522 ; 0.371 / 0.342 | 0.539 / 0.518 ; 0.389 / 0.368 | n.s. ; p = 0.27 / 0.17 | FT §12.4 |

**Answer:** learned judges move real formalization by 2–3 points at most on easy problems
(never significant on the granularity-robust word share); real Lean as a pruning oracle
moves it by +6 points (significant) at ~11 Lean calls per problem; trajectory-level
selection moves it by +6–8 points with CIs excluding zero.

## C. Why — the mechanism, measured

| finding | numbers | source |
|---|---|---|
| Offline AUC on FaithformBench does not transfer to the generator's own steps | 0.80–0.86 held-out → 0.50–0.61 vs real Lean on decoded steps, every encoder, every label source | §10.2 |
| Training on the generator's own Lean-labelled steps restores the judge | MiniLM 0.69 in-distribution (0.57–0.63 OOD); **Qwen3-1.7B 0.75 in-distribution, 0.72 / 0.65 / 0.74 / 0.71 OOD** (step-native MATH-500 / natural CoT / hard trees / hard beam steps) vs 0.55–0.65 for FaithformBench classifiers | FT §7.4, §8.2, §11.1 |
| Formalizability and correctness decouple on easy problems and couple on hard ones | pick-by-*true*-Lean-rate accuracy: easy cache 0.733 vs random 0.753 (ρ ≈ 0); hard trees **0.367 vs 0.308, ρ = +0.35**; MATH-train trees 0.386 vs 0.318, ρ = +0.25 | FT §4.1, §5.4, §11.1 |
| "Not proved" is search/statement failure, not falsity | 7 of 300 failures refutable by the prover (2.3%), 0/90 controls | FT §9 |
| The labels themselves depend on how the step is phrased for the autoformalizer | problem-in-view rendering formalized the *problem* for ~35% of steps; step-level rendering flips 33% of verdicts (66.6% agreement, proved 0.51 → 0.58) | §9.2, FT §4.1 |
| Binder statements were silently dropped by the pipeline | 607 steps (10–17% of every legacy cache) never reached the prover; 40.5% prove once they do; no arm ordering changes | FT §7.2, §9.4 |
| Step-capped decoding zeroes a large share of every arm | `max_steps=6`: 19–52% of trajectories end with an empty answer, unevenly across arms | §9 |
| The formalizer's own residual stream — and its attention heads — are usable but not better judges | FaithformBench-trained Goedel probes 0.58–0.64 on our steps (chance under the pipeline's rendering); probes refit on the generator's 4,435 labelled steps 0.67–0.78 OOD, = Qwen3 classifier on hard trees (0.778 vs 0.742), below it elsewhere (0.707 vs 0.720 easy; 0.583 vs 0.654 natural CoT); head probes with benchmark-selected heads = residual probe (±0.01); the head *selection* is the one FaithformBench artefact that transfers zero-shot (4 heads: 0.70 / 0.76 on easy / hard steps vs ≈ 0.6 for every benchmark-trained classifier) |
| Stronger steering saturates on formalization and trades accuracy | α 1 → 4 → 30: word share +2 → +2 → —, accuracy −3 → −9 → −10 points | FT §8.6 |

## D. Paper

**Working title.** *Lean-verified process rewards for mathematical reasoning: when
formalizability is (and is not) a proxy for correctness.*

**Thesis.** A step judge trained on real Lean verdicts can be made to transfer to a
generator's own reasoning (distribution first, capacity second), and real Lean in the
decoding loop reliably raises the verified content of solutions — but formalizability only
predicts correctness on hard problems, so the same judge that is a null on MATH-500
levels 1–3 selects correct solutions on levels 4–5.

1. **Introduction.** Process reward models from autoformalization: the promise (free,
   objective, unlimited labels) and the two things nobody has checked — whether the
   offline AUC transfers to the generator being decoded, and whether provability tracks
   correctness at all.
2. **Setup.** FaithformBench labels; MiniLM / Qwen3-1.7B judges; decoders (step rerank,
   constrained beam search, full-trajectory rerank, hybrid Lean-in-the-loop); the
   real-Lean pipeline (Goedel → DeepSeek-Prover-V2 → `lake repl`) run on fresh
   generations, with the three pipeline corrections as a methods contribution (§C rows
   5–7).
3. **Negative results on MATH-500 1–3** (Table A rows 1–11, Table B rows 1–6): no judge,
   strength, or Lean oracle moves accuracy; self-consistency does.
4. **The transfer gap and how to close it** (§C rows 1–2): FaithformBench-trained judges
   are at chance on decoded steps regardless of capacity; forked trajectory trees
   (43% Lean saving) give within-problem, in-generator labels; Qwen3-1.7B on 5,415 of
   them is the first judge that transfers (AUC 0.65–0.75 everywhere).
5. **Lean in the loop** (Table B row 7, FT §10): +6 points of verified content at
   unchanged accuracy for ~11 Lean calls/problem; the judge's role is call allocation.
6. **Where formalizability means correctness** (§C row 3, Table A rows 12–15): the
   oracle analysis across three problem sets, and the hard-problem rerank gain.
7. **What "proved" measures** (§C rows 4–6): refutation probe, rendering sensitivity,
   binder bug — the metric is "provable by this pipeline".
8. **Limitations.** 110/60/15-problem samples; one step-native generator whose SFT cost
   7 accuracy points; MATH-500 only; one seed; hard-problem result is a single split.
9. **Conclusion / what to do next.** Replicate §6 on a second hard sample and on the
   hard beam arms; Lean in the loop on hard problems; a judge trained on hard-problem
   trees; agreement-mode labels with the Qwen3 encoder.

**Venue fit.** As it stands: a strong analysis / negative-results paper (workshop or a
findings track) with one positive lead. If the hard-problem gain replicates on the
110-problem beam set and a second sample, it is a main-track methods-and-analysis paper.
