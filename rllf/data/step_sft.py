"""Builds the SFT dataset that teaches a base model to natively emit
`<step>` (and `<final>`) between reasoning steps -- see `segmentation_token.md`.

The whole point (per that document, point 4) is to teach the model the
*natural* segmentation language:

    step_1 <step> step_2 <step> ... step_n-1 <step> <final> step_n

(the doc's own worked example puts a short, separate "answer" line after
`<final>`, distinct from the last reasoning step; FaithformBench's step text
gives no such separate clean-answer field, so `render_step_sft_example`
below marks the trajectory's *last step itself* as the `<final>` segment
instead -- see its docstring for why, and what that trades off) by training
on complete, correctly-segmented trajectories, rather than
teaching it "write ONE step, then stop" (`DEFAULT_SYSTEM_PROMPT` in
`generation/base_model.py`) -- that instruction-based approach is the
generation-protocol mismatch `experiments_v1.md` §8 and
`lean-constrained-beam-search.md` §1 both point at.

FaithformBench's own unperturbed step spine (`rllf.data.faithform.load_steps`)
already *is* a naturally-segmented, correct trajectory per problem (real
model solutions, real ProcessBench-style segmentation-after-generation) --
no rendering choices of ours went into where those boundaries fall, which is
exactly the property this fine-tuning wants to imitate.

**Leakage boundary, non-negotiable:** this module only ever builds examples
from `split_trajectories_by_problem`'s *fit* half -- the same 20% of problems
`scripts/run_faithform.py`'s `stage_fit` trains reward models on, with the
same `(eval_fraction, seed)`. The 80% held-out half is what every AUC and
real-Lean number in this project's reports is measured against; this
fine-tuning must never see it, the same way `stage_fit` never trains the
reward model on it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..protocol import FINAL_TOKEN, STEP_TOKEN
from .trajectory import Trajectory

#: `segmentation_token.md` §3/§10's recommended prompt, verbatim in spirit:
#: simple, does not instruct "one step and stop", explicitly introduces both
#: new tokens and what each means.
SFT_SYSTEM_PROMPT = (
    "You are solving a mathematics problem.\n\n"
    "Produce a complete solution as a sequence of concise reasoning steps.\n"
    "Each step should express one coherent piece of reasoning.\n"
    f"Write {STEP_TOKEN} between consecutive reasoning steps.\n"
    f"Do not use {STEP_TOKEN} inside a reasoning step.\n"
    f"After the reasoning is complete, write {FINAL_TOKEN} followed by the final answer."
)


@dataclass
class StepSFTExample:
    problem_id: str
    prompt: str
    completion: str


def render_step_sft_example(problem_id: str, question: str, steps: Sequence[str]) -> "StepSFTExample | None":
    """One trajectory -> one (prompt, completion) SFT pair.

    `<step>` follows every step *except* the last; the last is always
    rendered under `<final>` instead. `<final>` means "the whole trajectory
    is complete, an answer follows" (segmentation_token.md) -- structurally
    that is always the trajectory's *last* segment, regardless of whether its
    text happens to already look like a clean answer. An earlier version of
    this function only used `<final>` when the last step passed
    `contains_final_answer` (contains `\\boxed{...}`/`####`/etc.) -- checked
    against the real data, that fired on only 2/227 fit-split examples,
    because FaithformBench's own step segmentation mostly does not preserve
    a literal answer marker on the final step even though it *is* the
    conclusion. That gave `<final>` essentially no training signal, so this
    marks the last step unconditionally instead, which also matches
    segmentation_token.md's own worked example (the last line is always
    under `<final>`, independent of its wording).

    Downstream answer extraction (`extract_final_answer`'s `FINAL_TOKEN`
    branch, `rllf/protocol.py`) takes the plain text after `<final>` verbatim
    when there is no `\\boxed{}`/`####` inside it; a prose-y last step (e.g.
    "Therefore, the total number of sticks they have is 34.") still resolves
    correctly via `NumericChecker`'s existing last-number fallback -- the
    same tolerance already relied on elsewhere in this codebase for prose
    final answers.

    Returns `None` for a trajectory with no usable (non-empty) steps.
    """
    cleaned = [s.strip() for s in steps if s.strip()]
    if not cleaned:
        return None
    *body, last = cleaned
    lines = [f"{s} {STEP_TOKEN}" for s in body]
    lines.append(f"{FINAL_TOKEN} {last}")
    prompt = f"{SFT_SYSTEM_PROMPT}\n\nProblem:\n{question.strip()}\n\nSolution:\n"
    return StepSFTExample(problem_id=problem_id, prompt=prompt, completion="\n".join(lines))


def split_trajectories_by_problem(
    trajectories: Sequence[Trajectory], eval_fraction: float, seed: int,
) -> Tuple[List[Trajectory], List[Trajectory]]:
    """The same by-problem split `scripts/run_faithform.py`'s `split_examples`
    uses for reward-model fitting, applied to whole trajectories instead of
    per-step examples -- pass the *same* `(eval_fraction, seed)` `stage_fit`
    used, and this reproduces exactly the same fit/held-out problem sets, at
    trajectory granularity. Returns `(fit, held_out)`."""
    problems = sorted({t.metadata["problem_id"] for t in trajectories})
    rng = random.Random(seed)
    rng.shuffle(problems)
    n_eval = int(round(len(problems) * eval_fraction))
    held = set(problems[:n_eval])
    fit = [t for t in trajectories if t.metadata["problem_id"] not in held]
    out = [t for t in trajectories if t.metadata["problem_id"] in held]
    return fit, out


def build_step_sft_examples(
    trajectories: Sequence[Trajectory], eval_fraction: float = 0.80, seed: int = 0,
) -> List[StepSFTExample]:
    """The fit-split-only SFT set. `eval_fraction`/`seed` must match whatever
    `stage_fit` used for the reward model this fine-tuned base model will be
    decoded with, so both components were trained on the identical fit/held
    boundary and neither ever saw the other's held-out data.
    """
    fit_trajectories, _held_out = split_trajectories_by_problem(trajectories, eval_fraction, seed)
    examples = []
    for t in fit_trajectories:
        example = render_step_sft_example(t.metadata["problem_id"], t.question, t.step_texts)
        if example is not None:
            examples.append(example)
    return examples
