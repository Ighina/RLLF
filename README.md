# RLLF — Reinforcement Learning from Lean Feedback

A **model-agnostic process reward model** for mathematical reasoning.

The base language model `M` is never fine-tuned. A separate, swappable decoding
policy `π_φ` is trained to steer generation toward reasoning steps that can be
**auto-formalized and verified in Lean**, and toward **correct final answers**:

```
M(q, u_<t)            ->  candidate continuations
π_φ(q, H_t, u_t)      ->  candidate scores
log p'(a) = log p_M(a) + α · log π_φ(a | state)
```

Two reward signals are kept deliberately separate, because a trajectory can be
locally verifiable yet wrong, and correct yet impossible to formalize:

| signal | source | value |
| --- | --- | --- |
| `r_step^t` | auto-formalize step `s_t`, check it with Lean | 1 if verified, else 0 |
| `r_outcome` | compare the final answer with a trusted reference | 1 if correct, else 0 |

## Install

```bash
pip install -e .            # torch + sympy
pip install -e ".[hf,dev]"  # + transformers (real base models) and pytest
```

Lean is optional: `elan`/`lean` on the `PATH` enables real verification, and
everything else falls back to `--mock-verifier`, which evaluates the same
extracted claims in exact rational arithmetic.

## Quickstart

```bash
# Everything end to end on synthetic problems (no downloads, ~20s)
python train.py demo --mock-verifier --oracle --ablation

# Or one stage at a time
python train.py generate --out runs/traj.jsonl --synthetic 80   # real Lean
python train.py train    --trajectories runs/traj.jsonl --out runs/prm
python train.py evaluate --policy runs/prm --synthetic 40 --oracle --ablation
```

With a real frozen base model:

```bash
python train.py generate --base-model Qwen/Qwen2.5-0.5B-Instruct \
    --problems examples/problems.jsonl --temperatures 0.7 1.0 --out runs/traj.jsonl
```

See `examples/huggingface_base_model.py` for the library API and
`examples/custom_verifier.py` for swapping in a domain-specific checker.

## The `<step>` protocol

The step boundary is part of the protocol, not formatting. `rllf/protocol.py`
is the single definition, imported by the dataset builder, verifier, decoder
and policy, so all four agree on where a reasoning unit ends:

```
Each of the 7 boxes holds 6 pens, so 7 * 6 = 42. <step>
After giving away 5 pens, 42 - 5 = 37. <step>
Final answer: 37
```

The final-answer segment is **not** a reasoning step: it goes to the outcome
checker and is never sent to the step verifier.

## Layout

```
rllf/
├── protocol.py              <step> boundaries, prompts, answer extraction
├── state.py                 PolicyState (q, H_t, u_t) and its rendering
├── pipeline.py              questions -> sampled trajectories -> labelled dataset D
├── data/
│   ├── trajectory.py        StepRecord / Trajectory, JSONL io, leak-free splits
│   ├── dataset.py           per-step states, preference pairs, listwise groups
│   └── synthetic.py         offline benchmark + a solver that makes real mistakes
├── generation/
│   ├── base_model.py        BaseModelAdapter, HFBaseModel (frozen), test doubles
│   └── guided_decoder.py    the guided decoding loop, all four modes
├── verification/
│   ├── formalizer.py        natural language -> Lean proposition
│   ├── lean_verifier.py     Lean subprocess, tactic ladder, cache, diagnostics
│   └── outcome_checker.py   exact / numeric / symbolic / solver / Lean checkers
├── policy/
│   ├── encoder.py           pretrained encoder, or a from-scratch fallback
│   ├── model.py             ProcessRewardModel: step, outcome, action, token heads
│   ├── scoring.py           PolicyAdapter, score combination, calibration
│   └── training.py          supervised, pairwise, listwise, policy gradient
├── evaluation/
│   ├── metrics.py           verification rate, calibration, AUC, cost
│   └── protocol.py          the six baselines and the alpha ablation
└── train.py                 CLI: generate / train / evaluate / demo
```

## 1. Data generation

`TrajectoryCollector` samples trajectories from **several generators** — models,
temperatures, prompting styles, seeds — then labels every step and final answer:

```python
collector = TrajectoryCollector(
    generators=[GeneratorSpec("qwen@T0.7", model, num_samples=4, temperature=0.7)],
    verifier=LeanStepVerifier(),
    outcome_checker=default_outcome_checker(),
)
trajectories = collector.collect(problems, output_path="runs/traj.jsonl")
```

Diversity is the point: a dataset drawn from one decoding configuration teaches
the policy to score *that configuration* rather than to score reasoning.

Each record is `{question, steps, final_answer, step_rewards, outcome_reward}`.

## 2. Step-level reward

```
natural-language step -> auto-formalization -> Lean -> r_step ∈ {0, 1}
```

`RuleBasedFormalizer` extracts the mathematical claim and renders it as a Lean
proposition. Two details matter:

* **It never simplifies.** `3 * 4 = 12` is passed through verbatim; letting a
  CAS normalise it first would turn the Lean check into a tautology and destroy
  the reward signal.
* **It picks the numeric type carefully.** `Nat` truncates subtraction and
  division, so any claim containing `-` or `/` is lifted to `Int`/`Rat`. Getting
  this wrong yields a statement that compiles but does not mean what the prose
  said — the failure mode of §14.

Claims with a free variable become implications, discharged by `omega`:

```
"Since x + 2 = 5, we get x = 3."   ->   ∀ (x : Nat), x + 2 = 5 → x = 3
"Therefore y = 8."  (history: "y - 3 = 5")   ->   ∀ (y : Int), y - 3 = 5 → y = 8
```

A variable with no defining hypothesis is reported as `unsupported_claim`
rather than universally quantified into a false statement.

The verifier emits one `example` per tactic in the ladder and reads the per-line
errors back, so the whole ladder costs **one** Lean process, not one per tactic.
Results are three-way, as specified, with richer diagnostics retained:

| status | reward | diagnostics |
| --- | --- | --- |
| `verified` | 1 | winning tactic, `kernel_checked`, duration |
| `formalization_failed` | 0 | `no_claim_found`, `unsupported_claim`, unbound variables |
| `lean_check_failed` | 0 | `false_proposition`, `unsolved_goal`, `parse_error`, `timeout`, … |

Operationally: results are cached on `sha256(question + previous_steps + step)`
(and invalidated when the checking regime changes), each compile runs in an
isolated temp directory under a timeout with deterministic settings, and sibling
candidates verify concurrently.

`norm_num` and friends are dropped automatically when mathlib is not on the
path. To use mathlib, point at a lake project:

```bash
python train.py generate --lean-project /path/to/project --lean-header "import Mathlib"
```

Without mathlib, `Rat` claims are closed by `native_decide`, which trusts the
compiler rather than the kernel — the `kernel_checked` diagnostic records this,
and `LeanConfig(allow_native_decide=False)` turns it off (at the cost of being
unable to check division at all, flagged as `uncheckable_regime`).

**[`LEAN_IMPLEMENTATION.md`](LEAN_IMPLEMENTATION.md) documents this layer in
full**: the exact subprocess contract, the one-file/many-tactics trick, the
diagnostic grammar, the cache key, the measured axiom footprint of each tactic,
and what the verifier deliberately does not do.

## 3. Outcome-level reward

`CompositeOutcomeChecker` tries exact match after normalisation, then numeric
equality, then symbolic equivalence; `CallableChecker` wraps a trusted solver
and `LeanOutcomeChecker` anchors the outcome to the same kernel as the steps.
Normalisation strips units, currency, thousands separators, `\boxed{}`, and
GSM8K-style `#### x`.

## 4. The policy

`ProcessRewardModel` is an encoder plus four heads:

| head | predicts | used by |
| --- | --- | --- |
| `step_head` | will this step verify in Lean? | reranking, rejection sampling |
| `outcome_head` | will this trajectory end correctly? | value estimate `V_φ(q, H_t, u_t)` |
| `action_head` | continue / close step / finish | learned step boundaries |
| `token_head` | per-token guidance scores | logit-level control |

The encoder is either a pretrained Hugging Face model (`--encoder
bert-base-uncased`) or a small from-scratch transformer over hashed tokens,
which lets the whole pipeline run with no downloads.

**Objectives** (`rllf/policy/training.py`), in the order to adopt them:

1. `train_supervised` — masked BCE on both heads,
   `L = BCE(r̂_step, r_step) + λ·BCE(r̂_outcome, r_outcome)`, with held-out
   checkpoint selection and early stopping.
2. `train_pairwise` — `L = -log σ(s(x⁺) − s(x⁻))` over verified/non-verified
   siblings of the *same* prefix; `train_listwise` ranks a whole candidate group,
   which is the decision the decoder actually makes.
3. `PolicyGradientTrainer` — REINFORCE over step selection with
   `G_t = Σ γ^{k-t} r_step^k + λ_outcome · r_outcome`, a moving baseline,
   advantage normalisation and an entropy bonus.

**Calibration.** A reward logit is not a log-probability. `TemperatureScaler`
fits Platt scaling on *held-out* states, is bounded so a degenerate fit cannot
collapse the policy into a constant, and is saved next to the weights.

## 5. Guided decoding

```python
decoder = GuidedDecoder(base_model, policy, verifier, outcome_checker,
                        DecodeConfig(mode="rejection", num_candidates=4, alpha=1.0))
result = decoder.generate(question, reference)
```

| mode | what it does |
| --- | --- |
| `plain` | baseline; the base model alone |
| `rerank` | sample K complete steps, keep the policy's favourite (**start here**) |
| `rejection` | rerank, verify the top candidates, resample on failure |
| `token` | per-token guidance, `log p' = log p_base + α · log π_φ` |

`α` is the guidance strength in both regimes: `α = 0` reproduces the base model
exactly; larger values sharpen the policy's influence. Both sides are
normalised to log-probabilities before combining, so `α` means the same thing
regardless of how the policy scales its raw scores.

**Verification schedule** (`verification=`): `online` (immediately at each
`<step>`), `deferred` (after the answer), `hybrid` (online only when the policy
is unsure — Lean calls are the expensive resource), or `none`.

**Boundary safeguards.** The learned action head never overrides the limits: a
step cannot close before `min_step_tokens` or run past `max_tokens_per_step`,
and on the last permitted step *close* becomes *finish* — it never truncates a
half-written step, which is the pathology the safeguards exist to prevent.

Token-level guidance works with any base model: `token_scoring="lookahead"`
scores the state each candidate token would produce, so the policy does not need
a shared vocabulary. `token_scoring="head"` is faster but does require
compatible tokenization — model-agnostic is not tokenizer-independent.

## 6. Evaluation

`run_evaluation` runs all six required comparisons on the same problems, with
every trajectory verified afterwards so process metrics are comparable even for
baselines that never call Lean during decoding. It reports outcome accuracy,
step verification rate, fully-verified rate, formalization success rate, tokens,
Lean calls, latency, resampling rate, plus per-difficulty breakdowns and reward
calibration (`binary_metrics` gives AUC, Brier, ECE).

A seventh, opt-in method (`--oracle`) uses the verifier *as* the policy. It is
not a baseline — it pays a Lean call per candidate — but it separates "the
reward model is weak" from "guided decoding is broken".

### Measured results: ProcessBench

The real experiment lives in
**[`PROCESSBENCH_EXPERIMENT.md`](PROCESSBENCH_EXPERIMENT.md)**: ProcessBench
problems, a frozen `Phi-3.5-mini-instruct`, real Lean verification, 40
validation / 158 held-out test problems split by question.

| method | answer accuracy | step verification | fully verified | s/problem |
| --- | --- | --- | --- | --- |
| `base_greedy` | 0.709 | 0.774 | 0.456 | 10.7 |
| `random_of_k` (compute-matched control) | 0.696 | 0.795 | 0.538 | 35.6 |
| **`policy_guided`** | **0.753** | **0.877** | **0.696** | 33.5 |

The process reward model reaches **0.956** held-out step AUC and transfers to
unseen problems (0.89-0.94, ECE 0.018-0.043). Guided decoding raises step
verification by +0.103 and fully-verified trajectories from 0.456 to 0.696,
**with no Lean calls at decode time**. Answer accuracy rises +0.044 but is not
significant (17-10, exact McNemar p = 0.248), and the outcome head does not
generalise (AUC 0.59-0.66). Read section 6.4 of that document before quoting any
of this.

```bash
python scripts/run_processbench.py --stage all --out runs/pb
```

### Measured results: synthetic smoke test

40 synthetic two-step problems, a base model with a 25 % per-step error rate,
mock verifier, `K = 4` candidates:

```
method                      outcome_accuracy  step_verification_rate  avg_lean_calls
greedy                                 0.550                   0.762           2.000
sampling                               0.525                   0.725           2.000
best_of_n                              0.450                   0.713           8.000
verification_only                      1.000                   1.000           3.900
policy_guided                          0.600                   0.762           2.000
policy_plus_verification               1.000                   1.000           3.350
oracle_policy                          1.000                   1.000           2.000
```

Read honestly:

* **Verification in the loop is what pays.** Rejection sampling against Lean
  reaches 1.000 for ~2 extra Lean calls per problem.
* **The decoding machinery is correct.** With a perfect process reward model
  (`oracle_policy`), reranking alone lifts 0.550 → 1.000 with *no* extra Lean
  calls at decode time. Guidance is doing what it is supposed to do.
* **The learned policy is the weak link here**, and deliberately so: the default
  encoder is a 2-layer transformer trained from scratch on ~2 000 states, and
  the synthetic task requires *arithmetic* judgement. It reaches ≈0.65 held-out
  AUC and ≈0.69 within-state pairwise accuracy, but `argmax` over candidates is
  far more brittle than pairwise accuracy — one confidently-wrong outlier (round
  numbers like `9 * 10 = 100` score high) is enough to pick the bad step. On
  some problem samples policy guidance beats the base model by a few points; on
  others it loses. With 40 problems that difference is inside the noise.

The `--ablation` flag sweeps `α`. If the policy is weak, accuracy *falls*
monotonically as `α` rises — that is the ablation working, and the reason to
verify the reward model before trusting it as a decoder.

For real use, train the policy on a pretrained encoder
(`--encoder <hf-model>`) and on trajectories from a real base model, and check
`step.auc` and `step.ece` before raising `α` above 0.

## 7. Extending it

Every boundary is an adapter:

```python
class BaseModelAdapter:      # HF, vLLM, or an API-backed model
    def next_token_logits(self, question, history, current_step): ...
    def generate_step_candidates(self, question, history, num_candidates): ...

class StepVerifier:          # Lean, a mock, or a domain rule
    def verify(self, question, previous_steps, current_step) -> StepVerification: ...

class OutcomeChecker:        # exact, numeric, symbolic, or a trusted solver
    def check(self, question, final_answer, reference) -> OutcomeVerification: ...

class PolicyAdapter:         # any scorer, learned or not
    def score_step_candidates(self, question, history, candidates): ...
    def score_tokens(self, state, token_ids): ...
```

## 8. Caveats carried into the code

* **A verifier reward is not a correctness reward.** A step can compile while
  formalizing the wrong claim, so the generated Lean statement is stored on
  every `StepRecord` for audit, and the numeric type is chosen conservatively.
* **Step boundaries are protocol**, defined once in `rllf/protocol.py`.
* **Lean is expensive**: cached, isolated, timed out, run concurrently.
* **Policy scores need calibration** before being treated as log-probabilities.
* **Model-agnostic ≠ tokenizer-independent**: token-level control wants shared
  tokenization; step-level reranking is what actually ports across models.
* **Splits are by question**, never by trajectory — the same question appears in
  many trajectories, and splitting on trajectories leaks the answer.

## Tests

```bash
python -m pytest -q     # 112 tests; Lean-dependent ones skip if lean is absent
```
