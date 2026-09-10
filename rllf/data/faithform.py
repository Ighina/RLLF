"""FaithformBench: real autoformalizer + prover verdicts as the step reward.

Each ProcessBench reasoning trace is decomposed into atomic steps; every step is
auto-formalized into Lean by eight different models and the resulting statement
is passed to a prover (DeepSeek-Prover-v2).  `proof_result` is the verdict.

    Faithform-AF.csv   goedel, kimina, herald, stepfun   (dedicated autoformalizers)
    Faithform-LLM.csv  gpt-5.2, gemini-3.1-pro, claude-opus-4-7, qwen-plus (frontier LLMs)

This replaces the rule-based formalizer entirely: no regex is involved in any
label produced here.

Two traps this module handles:

* **`node_id` is not a join key across the two files.**  The LLM file omits the
  steps classified `Declarative`, and its step numbering was recomputed after
  dropping them, so `step_5` denotes different text in each file.  The stable
  key is `(problem_id, step_content)`.
* **The AF file is the superset**, so its `node_id` order is what reconstructs
  the trace.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

AF_MODELS: Tuple[str, ...] = ("goedel", "kimina", "herald", "stepfun")
LLM_MODELS: Tuple[str, ...] = ("gpt-5.2", "gemini-3.1-pro", "claude-opus-4-7", "qwen-plus")
ALL_MODELS: Tuple[str, ...] = AF_MODELS + LLM_MODELS

#: Verdicts, normalised to upper snake case.  `PROVED` is the only positive.
PROVED = "PROVED"
REFUTED = "REFUTED"
DECLARATIVE = "DECLARATIVE"  # not a provable proposition (a definition/assignment)
FAILED = "FAILED"            # AF_FAIL / TC_FAIL / UNKNOWN aggregated

_FAILED_RAW = {"AF_FAIL", "TC_FAIL", "UNKNOWN"}

DEFAULT_ROOT = "~/Downloads/45134_FaithformBench_Benchmark_Code And Data Supplement 3"


def normalise_verdict(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip().upper().replace("-", "_")
    if text in (PROVED, REFUTED, DECLARATIVE):
        return text
    return FAILED if text in _FAILED_RAW else FAILED


def load_steps(root: str | Path = DEFAULT_ROOT, include_perturbed: bool = False):
    """Wide table: one row per (problem_id, step variant), one column per model.

    With `include_perturbed`, each node contributes up to two rows -- the true
    step and its `brokenmath-style_llm` corruption -- sharing a `step_index`, so
    they are alternative continuations of the same prefix.
    """
    import pandas as pd

    root = Path(os.path.expanduser(str(root)))
    cols = ["problem_id", "node_id", "is_perturbed", "autoformalizer_model", "proof_result", "step_content"]
    frames = []
    for name in ("Faithform-AF.csv", "Faithform-LLM.csv"):
        df = pd.read_csv(root / name, usecols=cols, low_memory=False)
        df["file"] = "AF" if "AF" in name else "LLM"
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["step_content"] = data.step_content.astype(str).str.strip()
    data["verdict"] = data.proof_result.map(normalise_verdict)

    # The AF file is the superset and carries the canonical ordering; the LLM
    # file's node_id is offset, so it is joined on text alone.
    af = data[data.file == "AF"].copy()
    af["order"] = af.node_id.str.extract(r"(\d+)").astype(float)

    truth = af[af.is_perturbed == False]  # noqa: E712
    spine = (
        truth.groupby(["problem_id", "node_id", "step_content"], as_index=False)["order"].min()
        .sort_values(["problem_id", "order"])
    )
    spine["step_index"] = spine.groupby("problem_id").cumcount()
    spine["is_perturbed"] = False

    if include_perturbed:
        corrupt = (
            af[af.is_perturbed == True]  # noqa: E712
            .groupby(["problem_id", "node_id", "step_content"], as_index=False)["order"].min()
        )
        corrupt = corrupt.merge(
            spine[["problem_id", "node_id", "step_index", "step_content"]]
            .rename(columns={"step_content": "true_content"}),
            on=["problem_id", "node_id"], how="inner",
        )
        # A "corruption" that left the text unchanged would carry both labels.
        corrupt = corrupt[corrupt.step_content != corrupt.true_content].drop(columns=["true_content"])
        corrupt["is_perturbed"] = True
        spine = pd.concat([spine, corrupt], ignore_index=True)

    # Text is the join key, so a text appearing at two nodes is ambiguous.
    counts = spine.groupby(["problem_id", "step_content"]).size()
    ambiguous = set(counts[counts > 1].index)
    spine = spine[~spine.set_index(["problem_id", "step_content"]).index.isin(ambiguous)]

    wide = data.pivot_table(
        index=["problem_id", "step_content"],
        columns="autoformalizer_model",
        values="verdict",
        aggfunc="first",
    ).reset_index()
    merged = spine.merge(wide, on=["problem_id", "step_content"], how="left")
    for model in ALL_MODELS:
        if model not in merged.columns:
            merged[model] = None
    return merged.sort_values(["problem_id", "step_index", "is_perturbed"]).reset_index(drop=True)


@dataclass
class LabelPolicy:
    """How a set of model verdicts becomes one binary step reward.

    `models`      -- which verdict columns to consult.
    `rule`        -- "single" | "majority" | "any" | "unanimous".
    `declarative` -- what a `Declarative` step is worth: "zero" (the
                     methodology's binary reward) or "mask" (excluded from the
                     step loss, on the grounds that "not a proposition" is a
                     statement about the step's *form*, not its correctness).
    `failed`      -- same choice for AF/typecheck/prover failures.
    """

    models: Tuple[str, ...] = ALL_MODELS
    rule: str = "majority"
    declarative: str = "zero"
    failed: str = "zero"
    name: str = ""
    #: "verifiability" (setting A) -- unperturbed steps only; 1 iff proved,
    #:   every other verdict 0.  Learns *can this step be formalized and proved*.
    #: "agreement" (setting B) -- keep a step only when the prover's verdict
    #:   matches the known ground truth: a true step that was PROVED (label 1)
    #:   or a corrupted step that was REFUTED (label 0).  Everything else is
    #:   discarded, so a surviving label is one whose formalization is
    #:   demonstrably faithful.  Learns *is this step correct*.
    mode: str = "verifiability"

    def label_row(self, verdicts: Sequence[Optional[str]], is_perturbed: bool) -> Optional[float]:
        if self.mode == "agreement":
            return self._agreement_label(verdicts, is_perturbed)
        return None if is_perturbed else self.label(verdicts)

    def _agreement_label(self, verdicts: Sequence[Optional[str]], is_perturbed: bool) -> Optional[float]:
        present = [v for v in verdicts if isinstance(v, str)]
        if not present:
            return None
        wanted = REFUTED if is_perturbed else PROVED
        n = sum(1 for v in present if v == wanted)
        agrees = {
            "single": n >= 1,
            "any": n >= 1,
            "majority": n * 2 > len(present),
            "unanimous": n == len(present),
        }[self.rule]
        if not agrees:
            return None  # verdict disagrees with ground truth -> unusable
        return 0.0 if is_perturbed else 1.0

    def label(self, verdicts: Sequence[Optional[str]]) -> Optional[float]:
        # `None` and pandas NaN both mean "this model has no verdict for this
        # step" -- the LLM file omits the Declarative steps entirely.  Counting
        # a missing verdict as "not proved" would label 18.6% of steps negative
        # for a reason that has nothing to do with the step.
        present = [v for v in verdicts if isinstance(v, str)]
        if not present:
            return None
        n_proved = sum(1 for v in present if v == PROVED)
        if n_proved:
            positive = {
                "single": n_proved >= 1,
                "any": n_proved >= 1,
                "majority": n_proved * 2 > len(present),
                "unanimous": n_proved == len(present),
            }[self.rule]
            if positive:
                return 1.0
        # Not positive.  Decide whether the negative is a usable label.
        if all(v == DECLARATIVE for v in present) and self.declarative == "mask":
            return None
        if all(v in (DECLARATIVE, FAILED) for v in present) and self.failed == "mask":
            return None
        return 0.0


def build_labels(table, policy: LabelPolicy):
    """Attach a binary `label` column; rows with no usable label get NaN."""
    import pandas as pd

    columns = [m for m in policy.models if m in table.columns]
    verdicts = table[columns].to_numpy(dtype=object)
    perturbed = (
        table["is_perturbed"].to_numpy() if "is_perturbed" in table.columns
        else [False] * len(table)
    )
    labels = [policy.label_row(row, bool(p)) for row, p in zip(verdicts, perturbed)]
    out = table.copy()
    out["label"] = pd.Series(labels, index=table.index, dtype="float64")
    return out


def to_trajectories(
    table,
    problem_text: Dict[str, str],
    exclude_problems: Optional[set] = None,
) -> List["Trajectory"]:
    """One `Trajectory` per problem, steps ordered, labels as step rewards."""
    from .trajectory import StepRecord, Trajectory

    exclude = exclude_problems or set()
    trajectories: List[Trajectory] = []
    for problem_id, group in table.groupby("problem_id", sort=True):
        if problem_id in exclude or problem_id not in problem_text:
            continue
        group = group.sort_values("step_index")
        records = [
            StepRecord(
                text=row.step_content,
                verified=bool(row.label == 1.0),
                reward=float(row.label) if row.label == row.label else -1.0,
                status="faithform",
                diagnostics={"labelled": bool(row.label == row.label)},
            )
            for row in group.itertuples()
        ]
        trajectories.append(
            Trajectory(
                question=problem_text[problem_id],
                steps=records,
                final_answer="",
                outcome_reward=1.0,  # every trace is an unperturbed, correct solution
                metadata={"problem_id": problem_id, "subset": problem_id.split("-")[0],
                          "source": "faithformbench"},
            )
        )
    return trajectories


def to_examples(
    table,
    problem_text: Dict[str, str],
    exclude_problems: Optional[set] = None,
) -> List["StepExample"]:
    """Labelled `(question, history, step)` states.

    The history is always the **true** prefix of the trace, never a corrupted
    one: at decode time the policy scores a candidate continuation of what has
    already been accepted.  A corrupted variant therefore shares its history
    with the true step it replaces, which is exactly the structure the pairwise
    and listwise objectives consume.
    """
    from .dataset import StepExample

    exclude = exclude_problems or set()
    examples: List[StepExample] = []
    for problem_id, group in table.groupby("problem_id", sort=True):
        if problem_id in exclude or problem_id not in problem_text:
            continue
        group = group.sort_values(["step_index", "is_perturbed"])
        truth = (
            group[~group.is_perturbed.astype(bool)]
            .sort_values("step_index")
            .step_content.tolist()
            if "is_perturbed" in group.columns
            else group.step_content.tolist()
        )
        question = problem_text[problem_id]
        for row in group.itertuples():
            if row.label != row.label:  # NaN -> no usable label
                continue
            index = int(row.step_index)
            examples.append(
                StepExample(
                    question=question,
                    history=truth[:index],
                    step=row.step_content,
                    step_reward=float(row.label),
                    outcome_reward=-1.0,  # every trace is correct: no outcome signal
                    step_index=index,
                    metadata={
                        "problem_id": problem_id,
                        "subset": problem_id.split("-")[0],
                        "is_perturbed": bool(getattr(row, "is_perturbed", False)),
                    },
                )
            )
    return examples


def processbench_problem_text() -> Dict[str, str]:
    from datasets import load_dataset

    dataset = load_dataset("Qwen/ProcessBench")
    return {row["id"]: row["problem"].strip() for split in dataset for row in dataset[split]}


def math500_problems():
    """MATH-500 as `Problem` objects, answers taken from the dataset."""
    from datasets import load_dataset

    from ..pipeline import Problem

    rows = load_dataset("HuggingFaceH4/MATH-500")["test"]
    return [
        Problem(
            question=r["problem"].strip(),
            reference=str(r["answer"]).strip(),
            metadata={"subject": r.get("subject", "?"), "level": r.get("level", "?"),
                      "unique_id": r.get("unique_id", ""), "difficulty": f"level-{r.get('level','?')}"},
        )
        for r in rows
    ]


def contaminated_problem_ids(problem_text: Dict[str, str]) -> set:
    """ProcessBench ids whose problem text also appears in MATH-500."""
    from datasets import load_dataset

    m500 = {r["problem"].strip() for r in load_dataset("HuggingFaceH4/MATH-500")["test"]}
    return {pid for pid, text in problem_text.items() if text in m500}
