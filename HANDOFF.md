# Handoff — what still needs running

State of the RLLF project as of 2026-09-10, and the experiments a remote
instance should pick up. Read `README.md` for the system,
`PROCESSBENCH_EXPERIMENT.md` for the first (superseded) experiment, and
`LEAN_IMPLEMENTATION.md` for the Lean layer.

---

## 0. TL;DR for whoever picks this up

* The **rule-based formalizer must not be used to produce experimental labels.**
  It is retained only for offline tests and CI. Every experimental label comes
  from FaithformBench (real autoformalizers + DeepSeek-Prover-v2).
* Two label settings are already fitted and compared (§2). The headline finding
  is that they measure *different things*.
* **MATH-500 decoding has not been run.** It is the main outstanding job (§3).
* One control was still running at handoff (§2.3) and may need restarting.

---

## 1. What is done

| artefact | where | note |
| --- | --- | --- |
| Library + tests | `rllf/`, `tests/` | 132 tests, `python -m pytest -q` |
| Lean verifier | `rllf/verification/` | real `lean` subprocess, no mathlib |
| ProcessBench experiment | `PROCESSBENCH_EXPERIMENT.md` | superseded; its reward was a regex artefact (§6.5 of that file) |
| FaithformBench loader | `rllf/data/faithform.py` | both label modes |
| Label-setting sweeps | `runs/ff/fit_report_{verifiability,agreement}.json` | 22 runs, MiniLM-L6 |
| Fitted policies | `runs/ff/prm_{mode}_{setting}/` | 22 checkpoints, **not in git** |

`runs/` is 2.3 GB and gitignored. Nothing in it is committed; everything is
reproducible from the scripts below.

### Data dependencies (not in git)

* **FaithformBench** — `Faithform-AF.csv`, `Faithform-LLM.csv`, expected under
  `~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3`
  (override with `--data-root`). ~365 MB.
* **ProcessBench** (`Qwen/ProcessBench`) and **MATH-500**
  (`HuggingFaceH4/MATH-500`) from the HF Hub; both were cached locally.
* Base model `microsoft/Phi-3.5-mini-instruct`; policy encoders
  `sentence-transformers/all-MiniLM-L6-v2` and `bert-base-uncased`.

---

## 2. The FaithformBench result (done — read before running anything)

Two ways to turn `proof_result` into a binary step reward:

* **A, `verifiability`** — unperturbed steps only; `1` iff PROVED, every other
  verdict `0`. Asks *"can this step be auto-formalized and proved?"*
* **B, `agreement`** — keep a step only when the verdict matches known ground
  truth: a true step PROVED (`1`) or a corrupted step REFUTED (`0`); discard
  everything else. Asks *"is this step correct?"* A surviving label is one whose
  formalization is demonstrably faithful.

### 2.1 Held-out AUC (80% split, MiniLM-L6)

| setting | A | B | Δ |
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

### 2.2 Reading

* **Mode A is learnable, mode B is near chance.** Formalizability is largely
  visible in a step's *form*; correctness is not.
* **Why:** 62.6% of corruptions differ **only in digits**, mean character
  similarity 0.97 (`6B + 7 = 37` → `5B + 7 = 37`). Detecting one requires doing
  the arithmetic. Classes are balanced in B, so this is not an imbalance effect.
* **Ensembling helps in A, hurts in B.** `all8_majority` is best in A (0.798,
  above every member) and worst in B (0.508): voting sharpens a real signal and
  merely discards data when there is none. Pairwise agreement between
  formalizers on the PROVED call is only 0.62–0.76.
* This retro-explains the earlier ProcessBench run, where a regex label reached
  0.956 AUC — also formalizability, and more trivially so.

### 2.3 Outstanding control (was running at handoff)

Does encoder capacity rescue mode B? `bert-base-uncased` on `agreement/gpt-5.2`
reached **0.582** by epoch 2 (MiniLM: 0.544), still far below mode A. Finish it:

```bash
python scripts/run_faithform.py --stage fit --out runs/ff --mode agreement \
  --settings gpt-5.2 --encoder bert-base-uncased --max-length 320 --epochs 8 --device cpu
```

> **Use `--device cpu`.** BERT-base on MPS OOM-killed twice on the original
> machine (~18 GB wired, at epoch 3) — a PyTorch MPS allocator issue, not real
> demand. On a machine with a CUDA GPU, `--device cuda` is fine.

---

## 3. MAIN OUTSTANDING JOB — MATH-500 decoding

Does policy-guided decoding raise **answer accuracy** on MATH-500?

```bash
python scripts/run_faithform.py --stage eval --out runs/ff \
  --sample 120 --policy-modes verifiability agreement \
  --policy-setting all8_majority --num-candidates 4 --alpha 1.0 \
  --max-steps 6 --max-new-tokens 72 --temperature 0.8
python scripts/run_faithform.py --stage report --out runs/ff
```

Five arms:

| arm | what it isolates |
| --- | --- |
| `base_greedy` | the frozen base model, one sample per step |
| `random_of_k` | same K candidates, chosen at random — compute-matched control |
| `self_consistency` | N whole solutions, majority vote on the answer |
| `policy_guided_verifiability` | mode A policy (AUC 0.798) |
| `policy_guided_agreement` | mode B policy (AUC ~0.51–0.54) |

The last two are the point: if A helps and B does not, the reward-model result
is confirmed behaviourally and not only in AUC.

**Practicalities**

* Measured on the original machine: `base_greedy` **14.8 s/problem**,
  `policy_guided` **56.6 s/problem**. Full 500 × 5 arms ≈ 26 h, hence
  `--sample 120` (stratified by difficulty level) ≈ 8 h. On a GPU box, raise
  `--sample` — prefer more problems over more arms.
* The run **checkpoints every 10 problems and resumes**; re-running skips
  completed arms. `--restart` forces a redo.
* `--policy-setting all8_majority` must name a directory that exists as
  `runs/ff/prm_<mode>_<setting>`. If `runs/` was not carried over, refit first:
  ```bash
  python scripts/run_faithform.py --stage fit --out runs/ff --mode verifiability \
    --encoder sentence-transformers/all-MiniLM-L6-v2 --max-length 320 --epochs 8
  python scripts/run_faithform.py --stage fit --out runs/ff --mode agreement \
    --encoder sentence-transformers/all-MiniLM-L6-v2 --max-length 320 --epochs 8
  ```
  (~10 min per setting, 22 settings.)

**Expected result, stated in advance.** `base_greedy` scored 0.25 on a
4-problem timing sample. If Phi-3.5-mini really sits near 25% on MATH-500, most
candidates at every step are wrong and reranking has little room to help. The
experiment may measure the base model's ceiling rather than the policy's value.
A stronger base model is the fix; say so in the write-up either way.

---

## 4. Then: `settings_comparison.md`

Requested deliverable, to be written once §2.3 and §3 have landed. It should
cover: what each setting measures; which formalizers are most learnable; whether
ensembling helps; whether A and B differ in *kind* rather than degree; and
whether the difference shows up downstream on MATH-500. Sections 2.1–2.2 above
are the raw material.

---

## 5. Traps already hit — do not rediscover these

| trap | consequence |
| --- | --- |
| `node_id` is **not** a join key across the two Faithform CSVs | the LLM file omits `Declarative` steps and renumbered; join on `(problem_id, step_content)` |
| pandas NaN is not `None` | missing verdicts silently became negative labels for 18.6% of steps |
| 30 "perturbed" steps are textually **identical** to their true counterpart | same string, contradictory labels — dropped |
| 40 `(problem_id, step_content)` keys map to >1 node | ambiguous text join — dropped |
| history must be the **true** prefix | a corrupted variant replaces step *k* but inherits true steps `0..k-1` |
| **46 MATH-500 problems appear in FaithformBench** | leakage; excluded from the training pool, benchmark kept whole |
| HF tokenizers truncate from the **right** | would cut off the current step, i.e. the thing being scored; `truncation_side="left"` is set in `build_encoder` |
| evaluating the full held-out split every epoch | cost 4× training; use `--monitor-size` |
| best-checkpoint snapshots kept on the accelerator | pinned a model copy per improving epoch; now snapshotted to CPU |

---

## 6. Environment notes

* macOS + MPS was the original host. MPS is fine for MiniLM, **not** for
  BERT-base (see §2.3). CUDA is untested here but the code paths are generic
  (`resolve_device` in `rllf/generation/base_model.py`).
* Lean 4.33.1 via elan, **no mathlib**. Only needed for the superseded
  ProcessBench experiment; the FaithformBench work needs no Lean at all.
* Offline runs: `HF_HUB_OFFLINE=1` with a warm HF cache.
* Verify the install before anything: `python -m pytest -q` (132 tests).
