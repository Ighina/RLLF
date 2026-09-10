"""The `<step>` protocol.

Section 14 of the methodology insists that the step boundary is part of the
protocol rather than cosmetic formatting: the dataset builder, the verifier,
the decoder and the policy must all agree on where one reasoning step ends and
the next begins.  Every component in this package imports its boundary logic
from here so that the convention is defined exactly once.

Wire format of a trajectory::

    Step 1 text <step>
    Step 2 text <step>
    Final answer: 42

`STEP_TOKEN` closes a reasoning step.  `FINAL_ANSWER_MARKER` opens the final
answer, which is *not* a reasoning step and is never sent to the step verifier.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

STEP_TOKEN = "<step>"
FINAL_ANSWER_MARKER = "Final answer:"

#: Tokens a tokenizer should learn as atomic units when one is extended.
SPECIAL_TOKENS: Tuple[str, ...] = (STEP_TOKEN,)

_FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|the\s+answer|answer)\s*(?::|=|\bis\b)\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_HASH_RE = re.compile(r"####\s*(.+)")


def contains_step_boundary(text: str) -> bool:
    """True once the running text has closed a step with `STEP_TOKEN`."""
    return STEP_TOKEN in text


def contains_final_answer(text: str) -> bool:
    """True once the running text has opened the final answer."""
    lowered = text.lower()
    return (
        FINAL_ANSWER_MARKER.lower() in lowered
        or "\\boxed{" in text
        or "####" in text
    )


def strip_step_token(text: str) -> str:
    """Remove the boundary token and surrounding whitespace from a step."""
    return text.replace(STEP_TOKEN, " ").strip()


def split_steps(text: str) -> Tuple[List[str], str]:
    """Split a rendered trajectory into ``(steps, final_answer)``.

    Text after the last `STEP_TOKEN` is treated as the final-answer segment.
    A trailing segment that carries no final-answer marker is still returned as
    the final answer so that partially formatted generations round-trip.
    """
    body, _, tail = text.rpartition(STEP_TOKEN)
    if not _:
        # No boundary token at all: everything is the final segment.
        return [], text.strip()
    steps = [s for s in (strip_step_token(p) for p in body.split(STEP_TOKEN)) if s]
    return steps, tail.strip()


def format_trajectory(steps: Sequence[str], final_answer: str) -> str:
    """Render steps + final answer back into the wire format."""
    parts = [f"{s.strip()} {STEP_TOKEN}" for s in steps if s.strip()]
    answer = final_answer.strip()
    if answer and not contains_final_answer(answer):
        answer = f"{FINAL_ANSWER_MARKER} {answer}"
    if answer:
        parts.append(answer)
    return "\n".join(parts)


def extract_final_answer(text: str) -> str:
    """Pull the answer payload out of a final-answer segment.

    Recognises ``Final answer: x``, ``\\boxed{x}`` and GSM8K-style ``#### x``.
    Falls back to the last non-empty line so that a caller always gets a string
    to hand to the outcome checker.
    """
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    hashed = _HASH_RE.findall(text)
    if hashed:
        return hashed[-1].strip().splitlines()[0].strip()
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        answer = match.group(1).strip()
        return answer.splitlines()[0].strip() if answer else ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def build_prompt(question: str, history: Iterable[str], current_step: str = "") -> str:
    """Render the generation prefix shown to a base model.

    The same rendering is used to build policy inputs, so a policy trained on
    offline trajectories sees exactly the strings the decoder produces.
    """
    lines = [f"Question: {question}", "Solution:"]
    lines.extend(f"{step.strip()} {STEP_TOKEN}" for step in history if step.strip())
    prefix = "\n".join(lines)
    return f"{prefix}\n{current_step}" if current_step else f"{prefix}\n"
