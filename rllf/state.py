"""The decoding state handed to the policy.

Shared by the dataset builder and the decoder so that a policy trained on
offline trajectories sees byte-identical strings at inference time.  Any
mismatch here silently degrades guidance, which is why the rendering lives in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from .protocol import FINAL_ANSWER_MARKER, build_prompt, contains_final_answer


@dataclass
class PolicyState:
    """`(q, H_t, u_t)`: question, step history, and the current partial step."""

    question: str
    history: List[str] = field(default_factory=list)
    current_step: str = ""
    final_answer: Optional[str] = None
    #: Opaque handle to the base model's internal state (KV cache, request id).
    base_model_state: Any = None

    def render(self) -> str:
        return render_state(self)

    def with_step(self, step: str) -> "PolicyState":
        return PolicyState(
            question=self.question,
            history=list(self.history),
            current_step=step,
            final_answer=self.final_answer,
            base_model_state=self.base_model_state,
        )

    def advanced(self, step: str) -> "PolicyState":
        return PolicyState(
            question=self.question,
            history=[*self.history, step],
            current_step="",
            base_model_state=self.base_model_state,
        )


def render_state(state: PolicyState) -> str:
    text = build_prompt(state.question, state.history, state.current_step)
    if state.final_answer:
        answer = state.final_answer.strip()
        if not contains_final_answer(answer):
            answer = f"{FINAL_ANSWER_MARKER} {answer}"
        text = f"{text}\n{answer}"
    return text


def render(question: str, history: Sequence[str], current_step: str = "", final_answer: Optional[str] = None) -> str:
    """Convenience wrapper for callers that do not hold a `PolicyState`."""
    return render_state(
        PolicyState(question=question, history=list(history), current_step=current_step, final_answer=final_answer)
    )
