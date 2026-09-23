# `lean_constrained_beam_search`

A new decoding mode implementing the "token beam search with delayed policy
evaluation" design from `update.md`, plus the calibration and intervention
machinery it needs. This document records what was built, the concrete
implementation choices made where the design left room, what is and is not
verified, and how to use it.

**Status: implemented, unit-tested (scripted/mocked base model), and now
also smoke-tested against a live model (5 problems, classifier mode; 2
problems, hybrid mode) — see §7 for what that run found, including one real,
useful negative result.**

---

## 1. What problem this solves

`update.md`'s core diagnosis, restated: `MODE_RERANK` samples K complete
`<step>`-token-truncated candidates independently, then throws away K−1 of
them. The classifier has no influence over *which prefixes the base model
explores* — if the good continuation never shows up among the K samples, no
reward model can recover it. Separately, forcing the base model to "write
ONE step, then stop" (`DEFAULT_SYSTEM_PROMPT` in `base_model.py`) is itself
an unnatural generation objective, and is a second, independent source of
the distribution mismatch this whole investigation has been chasing (see
`experiments_v1.md` §8's mechanism discussion, prompted by the same
observation this session confirmed by reading the code directly).

`lean_constrained_beam_search` addresses both: the base model does ordinary,
unconstrained token generation (`<step>` is a decoder-level *segmentation
event*, not an instruction the model is trained/prompted to obey), and beam
search — not independent K-sampling — keeps multiple partial trajectories
alive so the classifier's preference can actually steer *which prefixes get
explored further*, not just pick among whatever a single-shot sample
happened to produce.

## 2. What was implemented

| Piece | File | What |
| --- | --- | --- |
| The decoding mode itself | `rllf/generation/beam_search.py` (new file) | `lean_constrained_beam_search()`, `BeamSearchConfig`, `_Beam`, the three intervention functions |
| Base-model beam primitive | `rllf/generation/base_model.py` | `BaseModelAdapter.generate_step_beam` (abstract) + `HFBaseModel.generate_step_beam` (concrete, native HF beam search) |
| The classifier's beam-search score | `rllf/policy/scoring.py` | `ProcessRewardPolicy.beam_calibrator` (new `TemperatureScaler`, separate from `step_calibrator`/`outcome_calibrator`) + `predict_step_logit_for_beam()` |
| Fitting `tau_phi` | `rllf/policy/training.py` | `fit_beam_calibration()` |
| Persistence | `rllf/policy/scoring.py` | `save_policy`/`load_policy` gained a `"beam"` key in `calibration.json`, defaulting to the identity scaler so **every existing checkpoint still loads unchanged** |
| CLI to fit `tau_phi` | `scripts/run_faithform.py` | `--stage fit-beam-calibration` |
| Tests | `tests/test_beam_search.py`, additions to `tests/test_policy.py` | 9 + 6 new tests, all passing, all against scripted/mocked doubles (no real model) |

**Explicitly not touched, per the request:**
- The training target / loss for the reward model. `update.md` suggests
  retraining as a pairwise preference model; that suggestion was **not**
  implemented. `fit_beam_calibration` only fits a post-hoc scalar
  temperature+bias on top of the *existing* trained model's raw logit — it
  changes nothing about how the model itself is trained.
- `step_calibrator`/`outcome_calibrator` and everything that reads them
  (`rerank`, `self_consistency_weighted`, `stage_formalizability`, ...) —
  `beam_calibrator` is a new, separate attribute (see §4).
- Every existing decoding mode (`plain`, `rerank`, `rejection`, `token`) —
  `lean_constrained_beam_search` is additive, in a new file, reusing
  `DecodeResult`/`DecodeStats`/`_make_record` from `guided_decoder.py` but
  not modifying anything in it.

## 3. The re-scoring mechanism (request 1)

The classifier contributes **one scalar per completed step**, added directly
to the beam's cumulative score, exactly at the `<step>` boundary:

```python
beam.classifier_score += alpha * policy.predict_step_logit_for_beam(state)
```

No length normalisation of the classifier's contribution, and no
distributing it across the step's tokens — that per-token distribution is
what `MODE_TOKEN` already does (`guided_decoder._combine`), and is
deliberately not what this mode does. Length normalisation is applied only
to the *base model's* score, and only via HuggingFace's own `length_penalty`
parameter on the beam-search call (`update.md`'s β) — see §5.

## 4. Calibration (`tau_phi`) (requests 2, 2a)

```
f_phi(s) = logit(q_phi(s)) / tau_phi
```

`TemperatureScaler.apply_logits(logits) = (logits - bias) / temperature`
computes exactly this (`logit(sigmoid((z-b)/T)) == (z-b)/T`), so
`predict_step_logit_for_beam` reuses the *existing* `TemperatureScaler`
class rather than inventing a new one:

```python
# rllf/policy/scoring.py
def predict_step_logit_for_beam(self, state: PolicyState) -> float:
    scores = self._score_texts([state.render()])
    return float(self.beam_calibrator.apply_logits(scores["step_logit"])[0])
```

**Why a separate scaler from the existing `step_calibrator`, not a reuse of
it:** `step_calibrator` is fit to make `predict_step_probability` a
well-calibrated *classification* probability (the AUC/ECE numbers throughout
`experiments_v1.md`). `rerank`'s within-step `argmax` doesn't actually need
that calibration at all — any monotonic rescaling leaves an argmax over a
fixed candidate set unchanged, which is exactly why the "maybe it's a
calibration problem" hypothesis for the Qwen3-1.7B generalization gap turned
out to be provably wrong (`experiments_v1.md` §7). Beam search is different:
the classifier's score is **added** to the base model's cumulative
log-probability across a whole trajectory, so its absolute scale relative to
the LM's genuinely matters. `beam_calibrator` targets *that* problem — scale
matching for an additive combination — which is a different fitting target
even though the mechanics (`TemperatureScaler.fit`, LBFGS on masked BCE) are
identical. Keeping them as two separate attributes means fitting one can
never silently change the other's behaviour in `rerank` or anywhere else.

**Default value (request 2, "for now"):** `TemperatureScaler()` — `T=1,
bias=0`, i.e. the raw logit, unscaled. `save_policy`/`load_policy` always
write/read a `"beam"` key in `calibration.json`; a checkpoint saved before
this existed simply has no such key, and `load_policy` defaults it to the
identity rather than erroring (`test_load_policy_defaults_the_beam_calibrator_for_an_old_checkpoint`).

**Finding the real value (request 2a):**

```python
from rllf.policy.training import fit_beam_calibration
beam_scaler = fit_beam_calibration(model, calibration_examples)  # TrainConfig optional
```

or via the CLI, once a policy is already fit:

```bash
python3 scripts/run_faithform.py --stage fit-beam-calibration \
    --out /scratch/users/iacopog/rllf-runs/ff_qwen3-1.7b \
    --mode verifiability --policy-setting all8_majority --device cuda
```

This reads the *existing* `prm_verifiability_all8_majority` checkpoint
(unchanged), fits `tau_phi` on a split, and writes the result into that
checkpoint's `calibration.json` under `"beam"` — no retraining, no new
checkpoint directory.

**On the calibration split:** `stage_fit` already fits `step_calibrator`/
`outcome_calibrator` directly on the *whole* 80% held-out split — adequate
for a classification calibrator being reported on that same split's AUC/ECE,
but arguably too little separation for a second, differently-purposed
calibration. `stage_fit_beam_calibration` instead splits that held-out split
in half again (deterministically, via `seed + 1`): one half fits `tau_phi`,
the other is held out purely to report the classifier's (calibration-scaling
cannot change) AUC on data genuinely unseen by anything, as a sanity check
that the checkpoint and split are sound. This is a implementation choice
made to fill in a gap the request didn't fully specify (see §7) — flagged
explicitly rather than silently assumed.

## 5. The beam search algorithm (request 4)

The requested reference is specifically **"token beam search with delayed
policy evaluation"** in `update.md` — genuine token-level beam search
(continuous, LM-driven pruning *within* a step), with the classifier firing
*only* at `<step>` boundaries, at which point it can also prune. This is a
stronger requirement than the document's earlier, simpler "segment-level
beam search" sketch (each beam calls something like
`generate_step_candidates` independently in isolation) — the token-level
variant specifically wants real beam-search pruning to be already happening
during a step's own generation, not only at the boundary.

**Implementation strategy — round-based restarts of HuggingFace's own beam
search, not a hand-written token loop.** `model.generate(num_beams=...)` has
no hook to pause mid-search and inject an external score the instant some
(but not all) beams cross `<step>` — HF's beam search runs to completion in
one call. Reimplementing beam search primitives from scratch (padding, EOS
handling, length penalties, numerical stability) to get that hook was judged
higher-risk than the alternative actually implemented: run beam search in
**rounds**, one round per reasoning step.

```text
round k (parents = the up-to-`beam_width` survivors of round k-1):
  for each parent:
      children = model.generate(
          num_beams=expansion_per_beam, num_return_sequences=expansion_per_beam,
          stop_strings=["<step>"],
      )   # ← real, token-level beam search, HF's own internals: the "usual" pruning
  pool = union of every parent's children (+ any beams that already finished earlier)
  classifier/verifier/hybrid intervention fires on every newly-completed step  # ← the boundary
  beams = top `beam_width` of pool, by classifier-adjusted score              # ← the prune
```

- **Round 1** has exactly one parent (the empty history), so it requests
  `beam_width` children directly rather than `expansion_per_beam` — there is
  no sibling pool to over-provision against yet.
- **Every later round** over-provisions: `beam_width` parents ×
  `expansion_per_beam` children each = `beam_width * expansion_per_beam`
  pooled candidates, pruned back to `beam_width`. This over-provisioning is
  what gives the classifier *genuine* pruning power at the boundary — with
  `expansion_per_beam=1` there would be nothing to prune, only a re-ordering
  of who continues (still meaningful for the *final* answer's ranking, but
  not for steering exploration, which is the whole point per §1).
- Within one parent's `generate()` call, HF's own beam search does real
  token-level exploration and pruning using the base model's probabilities
  alone — the classifier never sees a token, only a completed step. This
  satisfies request 1's "does not require... distributing the classifier
  probability to all tokens" and keeps that concern (`MODE_TOKEN`) entirely
  separate.
- `length_penalty` (`BeamSearchConfig.length_penalty`) is HF's own
  GNMT-style normalisation of the *base model's* score within a step —
  `update.md`'s β. It is never applied to the classifier's contribution.

**`_Beam` bookkeeping:** each beam tracks `lm_score` and `classifier_score`
separately (summing to `.score`, what pruning and final selection use), so a
finished trajectory's score decomposes into "how much did the base model
like this" vs. "how much did the policy like this" — useful for exactly the
kind of analysis the rest of this investigation has been doing.

**Final-answer handling:** matches `guided_decoder._next_step_candidates`
exactly — a completed segment that looks like a final answer
(`contains_final_answer`) ends that beam, is *not* appended to
`history`/scored by the classifier or verifier, and is checked against the
reference by `outcome_checker` only at the very end, same as every other
mode.

### Known, deliberate simplification

A beam that finishes on round *k* is carried into later rounds' pooling
unmodified (no new LM or classifier score to add — it is done). A beam still
active in round *k+1* keeps accumulating both LM and classifier score. This
means a finished beam and an active beam are compared on a score built from
different numbers of "classifier opinions" — arguably fine (a final answer's
quality is a different kind of thing than a reasoning step's, and is checked
separately by the outcome checker after the fact), but worth stating plainly
rather than leaving implicit.

## 6. The three intervention modes (request 5)

`BeamSearchConfig.intervention`:

- **`"classifier"` (default).** `beam.classifier_score += alpha *
  policy.predict_step_logit_for_beam(state)` for every newly-completed step.
  No Lean calls at all.
- **`"verifier"`.** Every newly-completed step in the pool is sent to
  `verifier.verify_many` (grouped by shared parent history, so siblings from
  one expansion share a single batched call — the same batching
  `guided_decoder._rejection_sample` already uses). A verified step gets
  `verify_accept_bonus` (default `0.0`); a failing one gets
  `verify_reject_penalty` (default `-50.0`, large enough to be pruned out
  whenever any verified sibling survives, finite so an all-failing pool
  still ranks by *something* rather than every beam tying).
- **`"hybrid"`.** Mirrors `guided_decoder._maybe_verify_online`'s
  `VERIFY_HYBRID` branch, adapted from "should we bother recording a
  verification" to "which contribution should this beam's score get":
  compute the classifier's *classification*-calibrated confidence
  (`predict_step_probability`, i.e. `step_calibrator`, not
  `beam_calibrator`); if it's `>= hybrid_threshold`, trust the classifier
  alone (cheap path, same as `"classifier"` mode); otherwise escalate that
  step to a real Lean check and use the verifier's outcome instead (never
  both — they are on unrelated scales, so summing them would double-count
  in a way that has no principled scale).

## 7. What is and is not verified

**Verified (fast unit tests, `tests/test_beam_search.py` + additions to
`tests/test_policy.py`, all passing, no GPU or real model needed):**
- The classifier intervention correctly re-ranks and prunes across a round
  boundary, including when the base model's own preference disagrees with
  the classifier's.
- The classifier never fires on a final-answer completion.
- The verifier intervention rejects a step that fails real (well, mocked —
  `MockStepVerifier`'s deterministic arithmetic check) verification in
  favour of one that passes.
- The verifier/hybrid interventions raise a clear error rather than silently
  no-op-ing when no verifier is supplied.
- The hybrid intervention escalates *only* the step below its confidence
  threshold to the verifier, leaving the confident one scored by the
  classifier alone (checked via `stats.lean_calls`).
- The loop terminates at `max_steps`, and terminates early once every beam
  has independently found a final answer.
- `beam_calibrator` defaults to the identity, is independent of
  `step_calibrator`, round-trips through `save_policy`/`load_policy`, and
  defaults gracefully for a checkpoint saved before it existed.
- `fit_beam_calibration` returns a usable, fitted `TemperatureScaler`.

**Now verified, against a live model** (Phi-3.5-mini-instruct, the
un-fine-tuned base, + the fitted Qwen3-1.7B policy, `beam_width=3`,
`expansion_per_beam=2`, `max_steps=6`; smoke-test scripts and raw output in
`/scratch/users/iacopog/rllf-runs/beam_search_smoke/`):

- **`output.sequences_scores` is populated correctly** for this
  `num_beams`+`stop_strings`+`early_stopping=True` combination in the
  installed `transformers` version — every one of 5 classifier-mode problems
  and both hybrid-mode problems produced a non-zero `lm_score`. The `0.0`
  fallback exists but was never exercised in this run.
- **No crashes, sane per-call timing**: mean 1.04s/`generate_step_beam` call
  across 69 calls (5 problems), 14.1s/problem end to end. Hybrid mode's
  escalation logic fired correctly — both hybrid-mode problems recorded
  `lean_calls > 0` (5 and 9 respectively), confirming steps below the
  confidence threshold really do get sent to a real (`MockStepVerifier`)
  check rather than silently skipped.
- **A real, informative negative result, not a bug**: 4 of the 5
  classifier-mode problems ran the full `max_steps=6` without ever reaching
  a final answer (the fifth reached one but got truncated mid-expression by
  `max_tokens_per_step=128`). Reading the actual generated steps, this
  base model — not yet step-native — produces long, multi-clause segments
  per "step" (e.g. one whole sentence computing one variable of a
  multi-variable word problem), so a 6-step budget is consumed by genuine
  reasoning progress before an answer is reached, not by the search
  behaving badly. This is exactly the generation-protocol mismatch this
  document's §1 and `step-native-finetuning.md` both name as the underlying
  problem, showing up concretely rather than staying hypothetical. A
  follow-up run against the step-native fine-tuned checkpoint
  (`step-native-finetuning.md`) is in progress to check whether teaching the
  model to keep steps short changes this.

**Still not verified:**
- Whether `stop_strings` reliably stops **each beam independently** at its
  own `<step>` occurrence rather than only when *all* beams in a call have
  stopped — the smoke test's timings and per-call candidate counts are
  consistent with independent stopping, but this wasn't checked directly
  against `transformers`' internals.
- Behaviour at a larger `beam_width`/`expansion_per_beam`, or with the
  `verifier`-only intervention against a real (not mocked) Lean check —
  only `classifier` and `hybrid`-with-`MockStepVerifier` have been run live.
- The calibration split design in §4 (halving the existing held-out split)
  is a reasonable-but-unverified default; if a genuinely fresh, larger
  calibration set is available it would likely be better to use it via
  `fit_beam_calibration` directly (bypassing the CLI stage's split logic).
  `beam_calibrator` was left at its identity default throughout the smoke
  test above (not yet fit for this checkpoint).

## 8. Usage sketch

```python
from rllf.generation.base_model import GenerationConfig, HFBaseModel
from rllf.generation.beam_search import BeamSearchConfig, INTERVENE_HYBRID, lean_constrained_beam_search
from rllf.policy.scoring import load_policy
from rllf.verification.outcome_checker import default_outcome_checker
from rllf.verification.lean_verifier import LeanVerifier  # or MockStepVerifier for a dry run

base_model = HFBaseModel("Qwen/Qwen2.5-7B-Instruct", generation=GenerationConfig())
policy = load_policy(".../prm_verifiability_all8_majority", outcome_weight=0.0)
# after running `--stage fit-beam-calibration` once, `policy.beam_calibrator`
# is loaded automatically by `load_policy` -- nothing else to wire up.

result = lean_constrained_beam_search(
    question, reference, base_model, policy,
    verifier=LeanVerifier(...),  # only required for "verifier"/"hybrid"
    outcome_checker=default_outcome_checker(),
    config=BeamSearchConfig(
        beam_width=4, expansion_per_beam=2, alpha=1.0,
        intervention=INTERVENE_HYBRID, hybrid_threshold=0.5,
    ),
)
print(result.trajectory.step_texts, result.trajectory.final_answer)
print(result.trajectory.metadata["final_lm_score"], result.trajectory.metadata["final_classifier_score"])
```
