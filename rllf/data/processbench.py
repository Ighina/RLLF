"""ProcessBench as the source of real problems.

`Qwen/ProcessBench` collects model-written solutions to competition and
grade-school maths problems, each annotated with the index of the first
erroneous step (`label`, -1 when the solution is clean) and whether the final
answer is right (`final_answer_correct`).

We keep only the rows with ``final_answer_correct=True``.  Those are the rows
whose last step contains a *trusted* final answer, which is what the outcome
reward needs -- ProcessBench ships no separate ground-truth answer field.  The
annotated steps themselves are kept only as metadata: the experiment asks the
base model to produce its *own* solutions, and scores those.

    val  (20%)  ->  the model proposes solutions, Lean + the answer checker
                    score them, and the policy trains on the result
    test (80%)  ->  base model vs policy-guided decoding, held out
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..pipeline import Problem

SUBSETS: Tuple[str, ...] = ("gsm8k", "math", "olympiadbench", "omnimath")

_BOXED_OPEN = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_boxed(text: str) -> Optional[str]:
    """Last `\\boxed{...}`, matched with a brace counter.

    A regex cannot do this: MATH answers nest braces (`\\boxed{\\frac{1}{2}}`),
    and a non-greedy match would truncate at the first inner `}`.
    """
    found: Optional[str] = None
    for match in _BOXED_OPEN.finditer(text):
        start = match.end() - 1
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    found = text[start + 1 : index]
                    break
    return found


def reference_answer(steps: Sequence[str]) -> Tuple[Optional[str], str]:
    """Recover the trusted answer from a correct solution's final steps."""
    for step in reversed(steps):
        boxed = extract_boxed(step)
        if boxed and boxed.strip():
            return boxed.strip(), "boxed"
    if steps:
        numbers = _NUMBER.findall(steps[-1])
        if numbers:
            return numbers[-1].replace(",", ""), "last_number"
    return None, "none"


@dataclass
class ProcessBenchSplit:
    validation: List[Problem] = field(default_factory=list)
    test: List[Problem] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        from collections import Counter

        def describe(problems: Sequence[Problem]) -> Dict[str, Any]:
            return {
                "n": len(problems),
                "subsets": dict(Counter(p.metadata["subset"] for p in problems)),
                "answer_source": dict(Counter(p.metadata["answer_source"] for p in problems)),
            }

        return {"validation": describe(self.validation), "test": describe(self.test)}


def load_processbench(
    subsets: Sequence[str] = ("gsm8k",),
    val_fraction: float = 0.2,
    seed: int = 0,
    limit: Optional[int] = None,
    numeric_only: bool = False,
) -> ProcessBenchSplit:
    """Load, filter, deduplicate and split ProcessBench.

    Splitting is by *problem text*, not by row: ProcessBench contains several
    solutions to the same problem (375 distinct problems across 400 gsm8k rows),
    so a row-level split would put the same question on both sides.
    """
    import random

    from datasets import load_dataset

    dataset = load_dataset("Qwen/ProcessBench")
    by_question: Dict[str, Problem] = {}
    for subset in subsets:
        if subset not in dataset:
            raise ValueError(f"unknown ProcessBench subset {subset!r}; have {list(dataset)}")
        for row in dataset[subset]:
            if not row["final_answer_correct"]:
                continue
            answer, source = reference_answer(row["steps"])
            if answer is None:
                continue
            if numeric_only and not _is_numeric(answer):
                continue
            question = row["problem"].strip()
            if question in by_question:
                continue  # first correct solution wins
            by_question[question] = Problem(
                question=question,
                reference=answer,
                metadata={
                    "subset": subset,
                    "processbench_id": row["id"],
                    "reference_generator": row["generator"],
                    "answer_source": source,
                    "n_reference_steps": len(row["steps"]),
                    "difficulty": subset,
                    "reference_steps": list(row["steps"]),
                },
            )

    problems = sorted(by_question.values(), key=lambda p: p.metadata["processbench_id"])
    rng = random.Random(seed)
    rng.shuffle(problems)
    if limit is not None:
        problems = problems[:limit]
    n_val = int(round(len(problems) * val_fraction))
    return ProcessBenchSplit(validation=problems[:n_val], test=problems[n_val:])


def _is_numeric(answer: str) -> bool:
    try:
        float(answer.replace(",", "").replace("\\!", "").strip())
        return True
    except ValueError:
        return False
