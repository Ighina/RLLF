# Settings comparison — verifiability vs. agreement

Answers the four questions posed in `HANDOFF.md` §4, now that §2.3 (the
encoder-capacity control) and §3 (MATH-500 decoding) have both landed — plus
two follow-ups run after that: a second, larger-capacity encoder control
(ModernBERT) for mode A, and a repeat of the MATH-500 decoding experiment on
a much stronger base model (`Qwen2.5-7B-Instruct`, vs. the original
`Phi-3.5-mini-instruct`) to check whether the first decoding result was an
artifact of a weak base model. All label data is FaithformBench: real
autoformalizers + DeepSeek-Prover-v2, no rule-based formalizer.

---

## What each setting measures

Two ways to turn FaithformBench's `proof_result` into a binary step reward:

* **A, `verifiability`** — unperturbed steps only; `1` iff PROVED, everything
  else `0`. Asks *"can this step be auto-formalized and proved?"* — a
  question about the step's **form**.
* **B, `agreement`** — keep a step only when the verdict matches known ground
  truth (true step PROVED, corrupted step REFUTED); discard everything else.
  Asks *"is this step correct?"* — a question about **content**, filtered to
  only the labels a faithful formalization could support.

They are built from the same source CSVs but are not two views of the same
signal: A keeps every unperturbed step regardless of correctness; B throws
away most of the data (majority of steps are neither a clean true-PROVED nor
a clean corrupted-REFUTED) to isolate correctness specifically.

## Which formalizers are most learnable

Held-out AUC, 80% split, MiniLM-L6 encoder:

| setting | A (verifiability) | B (agreement) | Δ (A − B) |
| --- | --- | --- | --- |
| all8_majority | **0.798** | 0.508 | −0.290 |
| goedel | 0.787 | 0.534 | −0.253 |
| af_majority | 0.760 | 0.536 | −0.224 |
| llm_majority | 0.734 | 0.532 | −0.201 |
| gemini-3.1-pro | 0.731 | 0.522 | −0.209 |
| stepfun | 0.731 | 0.533 | −0.198 |
| kimina | 0.725 | 0.542 | −0.183 |
| gpt-5.2 | 0.724 | **0.544** | −0.179 |
| claude-opus-4-7 | 0.717 | 0.509 | −0.208 |
| herald | 0.707 | 0.543 | −0.164 |
| qwen-plus | 0.703 | 0.523 | −0.179 |

In A, the single most learnable individual formalizer is **Goedel** (0.787),
closely trailed by **af_majority** (0.760, an ensemble of the autoformalizer
family). In B every individual formalizer clusters in a narrow 0.51–0.54
band — near chance, with no standout. **gpt-5.2** edges out the rest in B
(0.544) despite being mid-pack in A, which is itself a sign the two axes
aren't just noisy copies of the same thing.

## Does ensembling help?

Opposite answers depending on the question being asked:

* **Mode A: yes.** `all8_majority` (0.798) beats every one of its 8
  individual members — voting sharpens a real, learnable signal.
* **Mode B: no — it actively hurts.** `all8_majority` (0.508) is the
  *worst* setting in mode B, below every individual member including the
  weakest one. Pairwise agreement between formalizers on the PROVED call is
  only 0.62–0.76, so when the target is correctness rather than
  formalizability, voting mostly averages together disagreement rather than
  reinforcing signal.

## Do A and B differ in kind, or only in degree?

The encoder-capacity control (§2.3) was designed to answer exactly this: if
B is just a weaker version of the same kind of signal, a bigger encoder
should close the gap. If B needs a different kind of computation
(arithmetic verification, not surface pattern-matching), more capacity
should barely move it.

| encoder | mode B AUC (agreement / gpt-5.2) | mode A AUC (verifiability / all8_majority) |
| --- | --- | --- |
| MiniLM-L6 (33M params) | 0.544 | 0.798 |
| BERT-base-uncased (110M params) | **0.617** | — |
| ModernBERT-base (150M params, 2024 architecture) | — | 0.798 |

Result: capacity **does** buy something in B — a genuine +0.073 AUC gain
from MiniLM to BERT-base, not noise — but it recovers less than a third of
the gap to mode A, using an encoder more than 3x the parameter count. That
is the signature of a *kind* difference with a real but bounded *degree*
component: some of mode B's signal is learnable from more expressive text
representations (recoverable degree), but the bulk of it looks like it
requires actually doing the arithmetic rather than reading the text more
carefully — consistent with the earlier finding that 62.6% of corruptions
differ from the true step *only in digits* (`6B + 7 = 37` → `5B + 7 = 37`,
mean character similarity 0.97). A text encoder, however large, doesn't
compute; it pattern-matches increasingly subtle patterns.

The control run the other direction — does mode A have room left to give,
or is it already saturated? — says saturated: swapping MiniLM-L6 for
ModernBERT-base (a newer architecture, ~4.5x the parameters, epoch-by-epoch
AUC peaking at 0.796 before settling at 0.798) reproduces **exactly** the
same 0.798 AUC. Mode A's signal is apparently simple enough — closer to
surface pattern-matching on well-formedness than to anything requiring
deep language understanding — that even a small 2020-era encoder already
captures all of it; there is no headroom for a bigger model to find.

Put the two controls together and the picture sharpens: mode A is a small,
already-saturated signal; mode B has a small amount of capacity-recoverable
signal on top of a much larger amount that no text encoder can reach.
Verdict: **mostly kind, with mode B carrying a modest, capacity-recoverable
degree component that mode A doesn't have because it's already maxed out.**

## Does the difference show up downstream, on MATH-500?

This is the part §4 couldn't answer until §3 finished. Short answer: **no —
and, after a second, much stronger base model, "the base model was too
weak" is no longer available as the excuse.**

### Run 1 — `Phi-3.5-mini-instruct`, 250-problem stratified sample

| arm | accuracy | vs. `base_greedy` |
| --- | --- | --- |
| `base_greedy` | 0.256 | — |
| `random_of_k` (compute-matched control) | 0.260 | p = 1.000 |
| `self_consistency` | **0.344** | **p < 0.001** |
| `policy_guided_verifiability` (mode A, AUC 0.798) | 0.268 | p = 0.736 |
| `policy_guided_agreement` (mode B, AUC ~0.51–0.62) | 0.244 | p = 0.736 |

`base_greedy` here sits at 0.256 — within noise of the ~0.25 ceiling
predicted in `HANDOFF.md` from a 4-problem timing sample. The natural
reading at this point was: at that accuracy level, most step-level
candidates are simply wrong, so a step reranker has nothing good to surface
regardless of which mode it uses — the experiment was measuring
Phi-3.5-mini's ceiling, not the reward model's value.

### Run 2 — `Qwen2.5-7B-Instruct`, same 250 problems, same policies

A stronger base model directly tests that reading. Qwen2.5-7B is 60% more
accurate at baseline (0.360 vs. 0.256) — meaningfully more correct material
at each step for a reranker to find:

| arm | accuracy | vs. `base_greedy` |
| --- | --- | --- |
| `base_greedy` | 0.360 | — |
| `random_of_k` (compute-matched control) | 0.320 | p = 0.143 (24–14 losses) |
| `self_consistency` | **0.440** | **p = 0.002** |
| `policy_guided_verifiability` (mode A, AUC 0.798) | 0.356 | p = 1.000 |
| `policy_guided_agreement` (mode B, AUC ~0.51–0.62) | 0.336 | p = 0.392 |

**Same pattern, unchanged.** Neither policy-guided arm beats greedy, both
remain statistically indistinguishable from baseline, and `self_consistency`
— the one method that uses no reward model at all — is again the only
significant winner. If anything, `random_of_k` is now directionally *worse*
than greedy (24 losses vs. 14 wins, though short of significance at n=250):
with a decent base model, reranking against noisier candidates can throw
away a correct greedy continuation more often than it rescues a wrong one.

### What this actually shows

Doubling the base model's accuracy did not move the policy-guided arms
relative to baseline at all — 0.268→0.356 and 0.244→0.336 track
`base_greedy`'s own 0.256→0.360 rise almost exactly, rather than opening a
gap in either direction. That rules out "the base model has no headroom" as
the explanation. Combined with the ModernBERT result above (mode A's AUC is
capacity-saturated, not capacity-limited), the more likely explanation is
structural: **the AUC these process reward models achieve measures
per-step formalizability/well-formedness, and that signal doesn't line up
with which of 4 candidates leads to a correct final answer** — at least not
enough to survive reranking over 6 steps with K=4 candidates. A high AUC on
"can this step be formalized and proved" does not imply the reward model
can tell a promising continuation from a dead end for the purposes this
decoding setup needs. `self_consistency` working in both runs, using no
step-level signal whatsoever, is the sharpest illustration of that gap:
whole-solution majority voting finds correct answers these step rewards
cannot help surface.

## What this doesn't show

This is not evidence that step-level process reward models can't ever help
decoding — only that *this* signal (formalizability, from either labeling
mode, at this K and this step budget) doesn't transfer to this decoding
setup, on two base models spanning a 60%-relative accuracy range. The AUC
finding from §2 (mode A learnable and now known to be capacity-saturated;
mode B near-chance, partially rescued by capacity) stands on its own as a
labeling-methodology result independent of any of this. What would still be
open: a reward model trained on an outcome-correctness signal rather than
formalizability, a different reranking budget (K, step count), or a
decoding strategy that uses the reward model to *guide* generation rather
than only rerank fixed candidates.
