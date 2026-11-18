# Next steps: a benchmark of verifier-routed reasoning steps

Status: proposal, 2026-11-18. First annotation tool: `tools/verifier_annotation/annotate.html`.

## 1. Why build it

Our experiments keep meeting the same missing object. We can ask *whether a step is right* with three kinds of
checker (Lean, Python, a decision model with evidence), and each succeeds on a different kind of step. What we
cannot yet measure is **which checker a step calls for**, because no dataset says so:

- The heuristic label we used (python if the Python check verified, lean if Lean proved, none otherwise) describes
  what *our pipeline managed*, not what the step *is*. Half of all steps land in "none" through checker failure.
- A zero-shot semantic router (OpenJev: formalisable / computational / unverifiable) does not predict which
  verifier succeeds (AUC 0.51–0.63, §13.11 of the retraining notes). Either the semantic categories are wrong, or
  the checkers are too weak to realise them. Only human labels can separate the two.
- Existing step-level benchmarks label *correctness* (ProcessBench, PRM800K) or *faithfulness to evidence*
  (EntailmentBank, FinChain's executable traces), never the *mode of verification*.

A benchmark of steps annotated with (a) the adequate verifier and (b) correctness would let anyone evaluate a
router, a verifier, or the pair, independently of our particular pipeline.

## 2. What one item looks like

```json
{"trace": "finqa-0412", "source": "FinQA / Qwen2.5-7B", "task": "...", "context": "<table + text>",
 "step": 3, "text": "Revenue grew from 4.2bn to 5.1bn, i.e. by 21.4%.",
 "verifier": "python", "correct": "correct", "note": "", "machine": {"python": "verified"}}
```

**Verifier set (v0, five labels).**

| label | the step is… | checked by |
|---|---|---|
| `python` | computable: arithmetic, algebra on given values, counting, unit conversion, table lookups with arithmetic | a short program |
| `lean` | a general mathematical or logical claim: identity, inequality, property, deduction | autoformalization + prover |
| `evidence` | a factual claim attributable to supplied or retrievable documents | entailment against evidence (a decision model such as Jev, or NLI) |
| `judge` | a judgement no mechanical check gives: interpretation of the question, a modelling choice, a commonsense inference | a model or human judge |
| `none` | nothing to verify: plan, definition, restatement, assumption | — |

Open design questions to settle in the pilot: whether a step may carry two labels (a table lookup *and* a
computation), whether `lean` should be split into "formalisable in principle" and "formalisable with current
libraries", and whether to label the *minimal* verifier (cheapest that suffices) or *all* adequate ones.

## 3. Data sources

The aim is traces that mix reasoning modes inside one solution, from models, not gold solutions, so that wrong
steps are present. Candidates, in the order I would use them:

| source | reasoning modes it exercises | on this machine | notes |
|---|---|---|---|
| **FinQA / TAT-QA / ConvFinQA** | evidence (table + text lookup) → python (arithmetic) → judge (which figure the question means) | no | the best single source for mixed evidence + computation |
| **FinChain** | python, with domain formulas; gold executable traces | yes (cloned; fresh instances can be generated) | contamination-free; good for python vs none |
| **HotpotQA / 2WikiMultihopQA / MuSiQue** | evidence, multi-hop, with gold supporting facts | HotpotQA yes | gold supporting facts give free machine labels for `evidence` |
| **MATH / Omni-MATH / AIME** | python and lean | yes | we already hold 21k steps with Python and Lean verdicts |
| **StrategyQA / EntailmentBank** | judge and evidence, implicit decomposition | no | EntailmentBank has gold entailment trees |
| **ScienceQA / GPQA / TheoremQA** | lean-like derivations + python + evidence (formula recall) | ScienceQA, GPQA yes | physics and chemistry steps test the `judge` boundary |
| **Agentic / tool traces (SWE-style, spreadsheets)** | python (executable), evidence (file contents), judge (plan) | SpreadsheetBench, SWE-Skills-Bench cached | later phase; long traces |

Generators: at least two model families and two sizes per source, so the benchmark is not one model's style.
Sampling: stratify by source, by outcome (correct / wrong final answer) and by machine verdict, and over-sample
steps where the machine verdicts disagree (Python verified but Lean failed, and the reverse).

## 4. Annotation protocol

1. **Pre-labels.** Every step gets the machine verdicts we can already compute (Python check, Lean pipeline, Jev
   with evidence). They are shown to the annotator as hints, not as answers, and stored separately.
2. **Two annotators per step**, a third on disagreement. Report Cohen's kappa per label; the pilot's first
   deliverable is whether the five labels can be applied consistently at all (target kappa ≥ 0.6).
3. **Pilot:** 60 traces (about 500 steps) from four sources, both of us annotating all of it, then revise the label
   guide. **Phase 1:** 500 traces (about 4,000 steps). An LLM-assisted pass can propose labels afterwards, but only
   once the pilot gives a human-agreement ceiling to compare it with.
4. **Correctness** is labelled alongside (`correct / wrong / unsure`) so the benchmark also supports the
   question "does routing to the adequate verifier catch more wrong steps".

## 5. What the benchmark would evaluate

- **Routing accuracy:** predict the verifier label from (task, previous steps, step). Baselines we already have:
  the OpenJev zero-shot router, the Goedel-state probe, the fine-tuned generator's own tags.
- **Verifier quality given the right route:** for steps labelled `python`, how often does a code-model checker
  decide them, and how often correctly; likewise Lean and evidence. This separates router error from checker
  weakness, which our current labels cannot do.
- **End-to-end:** wrong-step detection (AUROC, and recall at a fixed false-alarm rate) for a routed system
  against any single verifier applied to everything.

## 6. Annotation tool (first implementation)

`tools/verifier_annotation/annotate.html` is a single self-contained page: open it in a browser, no server.

- Loads the embedded sample (8 MATH-500 model traces with Python verdicts as hints, 4 FinChain gold traces) or
  any `.json` / `.jsonl` file of `{id, source, task, context, final_answer, steps: [text | {text, suggested, machine}]}`.
- Per step: one verifier label, one correctness label, a free-text note. Machine verdicts and the suggestion are
  displayed but never pre-selected.
- Keyboard: `1`–`5` verifier, `c / w / u` correctness, `j / k` next and previous step.
- Progress is kept in the browser's local storage; **Export** writes JSON Lines, one object per annotated step,
  into a text box to copy out.

Known gaps for a real campaign: annotator identity and multi-annotator merge, a shared store instead of local
storage, adjudication view, and a larger rendering area for tables (FinQA) and long evidence.

## 7. Cost and risks

- Pilot: two people, about 500 steps each, roughly a day each with the tool.
- Phase 1 at two annotations per step: about 8,000 judgements. Machine pre-labels cost under 50 GPU-hours.
- Main risk: the labels are not separable in practice (many steps are both `python` and `lean`). The pilot is
  designed to find that out cheaply; multi-label annotation is the fallback.
- Second risk: a benchmark defined by *our* five verifiers ages with them. Mitigation: label the mode of
  verification, not the tool, and keep the tool-specific verdicts as separate machine fields.
