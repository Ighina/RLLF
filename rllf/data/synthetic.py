"""A synthetic arithmetic benchmark and base model.

Real trajectories should come from a real base model, but the pipeline needs
something deterministic to exercise in tests, demos and CI.  These word
problems have two arithmetic steps and an exact reference answer, and the
synthetic solver makes mistakes at a configurable rate -- so the resulting
dataset contains both verifiable and non-verifiable steps, which is exactly the
signal the process reward model is meant to learn.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..generation.base_model import BaseModelAdapter
from ..protocol import FINAL_ANSWER_MARKER
from ..pipeline import Problem


@dataclass
class _Template:
    name: str
    question: str
    first_step: str
    second_step: str


_TEMPLATES: Tuple[_Template, ...] = (
    _Template(
        "mul_then_sub",
        "A shop has {a} boxes of pens with {b} pens in each box. It gives away {c} pens. How many pens are left?",
        "Each of the {a} boxes holds {b} pens, so {a} * {b} = {m}.",
        "After giving away {c} pens, {m} - {c} = {r}.",
    ),
    _Template(
        "add_then_mul",
        "Ana walks {a} km on Monday and {b} km on Tuesday. She repeats that pattern for {c} weeks. How many km does she walk?",
        "In one week she walks {a} + {b} = {m}.",
        "Over {c} weeks that is {m} * {c} = {r}.",
    ),
    _Template(
        "div_then_add",
        "A baker splits {a} rolls equally into {b} trays, then adds {c} extra rolls to one tray. How many rolls are on that tray?",
        "Splitting {a} rolls into {b} trays gives {a} / {b} = {m}.",
        "Adding {c} more rolls, {m} + {c} = {r}.",
    ),
)


def make_problems(n: int = 24, seed: int = 0) -> List[Problem]:
    """Generate `n` two-step word problems with exact references."""
    rng = random.Random(seed)
    problems: List[Problem] = []
    for index in range(n):
        template = _TEMPLATES[index % len(_TEMPLATES)]
        values = _sample_values(template, rng)
        problems.append(
            Problem(
                question=template.question.format(**values),
                reference=str(values["r"]),
                metadata={
                    "template": template.name,
                    "difficulty": "easy" if values["r"] < 50 else "hard",
                    **{k: v for k, v in values.items() if k in "abcmr"},
                },
            )
        )
    return problems


def _sample_values(template: _Template, rng: random.Random) -> Dict[str, int]:
    if template.name == "mul_then_sub":
        a, b = rng.randint(3, 9), rng.randint(4, 12)
        m = a * b
        c = rng.randint(2, max(3, m // 2))
        return {"a": a, "b": b, "c": c, "m": m, "r": m - c}
    if template.name == "add_then_mul":
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 6)
        m = a + b
        return {"a": a, "b": b, "c": c, "m": m, "r": m * c}
    b = rng.randint(2, 6)
    m = rng.randint(4, 15)
    a = b * m
    c = rng.randint(1, 9)
    return {"a": a, "b": b, "c": c, "m": m, "r": m + c}


class SyntheticBaseModel(BaseModelAdapter):
    """A scripted solver that makes arithmetic slips at a controlled rate.

    Its second step is computed from whatever value the *history* contains, so
    an early mistake propagates the way a real model's would -- which is what
    makes the outcome reward differ from the step rewards.
    """

    def __init__(
        self,
        problems: Sequence[Problem],
        error_rate: float = 0.35,
        seed: int = 0,
        style: str = "plain",
    ):
        self.by_question: Dict[str, Problem] = {p.question: p for p in problems}
        self.error_rate = error_rate
        self.style = style
        self.rng = random.Random(seed)

    @property
    def tokenizer(self) -> Any:  # pragma: no cover - candidate-level model
        raise NotImplementedError("SyntheticBaseModel generates whole steps")

    def _template(self, problem: Problem) -> _Template:
        name = problem.metadata.get("template")
        for template in _TEMPLATES:
            if template.name == name:
                return template
        return _TEMPLATES[0]

    def _perturb(self, value: int, rng: Optional[random.Random] = None) -> int:
        delta = (rng or self.rng).choice([-3, -2, -1, 1, 2, 10])
        return max(0, value + delta)

    def generate_step_candidates(
        self, question: str, history: Sequence[str], num_candidates: int = 4, **kwargs: Any
    ) -> List[str]:
        problem = self.by_question.get(question)
        if problem is None:
            return []
        # Greedy decoding must be deterministic for a given state, the way a
        # real argmax decoder is -- so the greedy baseline is reproducible and
        # wrong on a stable subset of problems rather than a fresh one each run.
        rng = self.rng
        if kwargs.get("do_sample") is False:
            rng = random.Random(f"{question}|{len(history)}|{self.error_rate}")
        template = self._template(problem)
        values = {k: int(problem.metadata[k]) for k in "abcmr" if k in problem.metadata}
        index = len(history)
        candidates: List[str] = []
        for _ in range(num_candidates):
            if index == 0:
                m = values["m"]
                if rng.random() < self.error_rate:
                    m = self._perturb(m, rng)
                candidates.append(_style(template.first_step.format(**{**values, "m": m}), self.style))
            elif index == 1:
                m = _last_number(history[-1], default=values["m"])
                result = _apply_second_operation(template, m, values["c"])
                if rng.random() < self.error_rate:
                    result = self._perturb(result, rng)
                candidates.append(
                    _style(template.second_step.format(**{**values, "m": m, "r": result}), self.style)
                )
            else:
                answer = _last_number(history[-1], default=values["r"])
                candidates.append(f"{FINAL_ANSWER_MARKER} {answer}")
        return candidates

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return ""


def _style(text: str, style: str) -> str:
    """Vary surface form so the dataset is not single-style."""
    if style == "terse":
        return re.sub(r"^[^,]*,\s*", "", text)
    if style == "verbose":
        return f"Let us compute this carefully. {text}"
    return text


def _apply_second_operation(template: _Template, m: int, c: int) -> int:
    if template.name == "mul_then_sub":
        return m - c
    if template.name == "add_then_mul":
        return m * c
    return m + c


def _last_number(text: str, default: int = 0) -> int:
    numbers = re.findall(r"-?\d+", text)
    return int(numbers[-1]) if numbers else default
