"""RLLF -- Reinforcement Learning from Lean Feedback.

A model-agnostic process reward model for mathematical reasoning.  The base
language model is never fine-tuned; a separate decoding policy `pi_phi` is
trained to steer generation toward steps that auto-formalize and verify in
Lean, and toward correct final answers.
"""

from .protocol import (
    FINAL_ANSWER_MARKER,
    STEP_TOKEN,
    build_prompt,
    extract_final_answer,
    format_trajectory,
    split_steps,
)
from .state import PolicyState, render_state

__all__ = [
    "STEP_TOKEN",
    "FINAL_ANSWER_MARKER",
    "build_prompt",
    "extract_final_answer",
    "format_trajectory",
    "split_steps",
    "PolicyState",
    "render_state",
    "__version__",
]

__version__ = "0.1.0"
