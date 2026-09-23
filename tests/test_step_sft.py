import pytest

from rllf.data.step_sft import (
    SFT_SYSTEM_PROMPT,
    build_step_sft_examples,
    render_step_sft_example,
    split_trajectories_by_problem,
)
from rllf.data.trajectory import StepRecord, Trajectory
from rllf.protocol import FINAL_TOKEN, STEP_TOKEN


def _traj(problem_id, steps, question="q"):
    return Trajectory(
        question=question,
        steps=[StepRecord(text=s) for s in steps],
        final_answer="",
        metadata={"problem_id": problem_id},
    )


def test_render_joins_body_steps_with_the_step_token_and_marks_the_last_as_final():
    # The last step is always the `<final>` segment -- regardless of its
    # wording -- not conditional on it already looking like a clean answer
    # (see the docstring: that heuristic fired on only 2/227 real examples).
    example = render_step_sft_example("p1", "q", ["a", "b"])
    assert example.completion == f"a {STEP_TOKEN}\n{FINAL_TOKEN} b"
    assert example.prompt.startswith(SFT_SYSTEM_PROMPT)
    assert "Problem:\nq" in example.prompt


def test_render_marks_the_last_step_as_final_even_when_prose_not_a_clean_answer():
    example = render_step_sft_example("p1", "q", ["a", "so the total is 6 apples"])
    assert example.completion == f"a {STEP_TOKEN}\n{FINAL_TOKEN} so the total is 6 apples"


def test_render_strips_whitespace_and_drops_empty_steps():
    example = render_step_sft_example("p1", "q", ["  a  ", "", "   ", "b"])
    assert example.completion == f"a {STEP_TOKEN}\n{FINAL_TOKEN} b"


def test_render_returns_none_for_no_usable_steps():
    assert render_step_sft_example("p1", "q", ["", "   "]) is None
    assert render_step_sft_example("p1", "q", []) is None


def test_split_trajectories_by_problem_puts_no_problem_on_both_sides():
    trajectories = [_traj(f"p{i}", ["a", "b"]) for i in range(20)]
    fit, held = split_trajectories_by_problem(trajectories, eval_fraction=0.8, seed=0)
    fit_ids = {t.metadata["problem_id"] for t in fit}
    held_ids = {t.metadata["problem_id"] for t in held}
    assert not (fit_ids & held_ids)
    assert fit_ids | held_ids == {f"p{i}" for i in range(20)}
    assert len(held) == pytest.approx(16, abs=1)  # 80% of 20


def test_split_trajectories_by_problem_is_deterministic():
    trajectories = [_traj(f"p{i}", ["a"]) for i in range(10)]
    fit1, held1 = split_trajectories_by_problem(trajectories, eval_fraction=0.8, seed=7)
    fit2, held2 = split_trajectories_by_problem(trajectories, eval_fraction=0.8, seed=7)
    assert [t.metadata["problem_id"] for t in fit1] == [t.metadata["problem_id"] for t in fit2]
    assert [t.metadata["problem_id"] for t in held1] == [t.metadata["problem_id"] for t in held2]


def test_build_step_sft_examples_only_draws_from_the_fit_split():
    trajectories = [_traj(f"p{i}", ["a", "b"]) for i in range(20)]
    _fit, held = split_trajectories_by_problem(trajectories, eval_fraction=0.8, seed=0)
    held_ids = {t.metadata["problem_id"] for t in held}

    examples = build_step_sft_examples(trajectories, eval_fraction=0.8, seed=0)
    example_ids = {e.problem_id for e in examples}

    assert not (example_ids & held_ids)  # the "test part" is never touched
    assert len(examples) == 20 - len(held_ids)


def test_build_step_sft_examples_skips_trajectories_with_no_usable_steps():
    trajectories = [_traj("p0", ["a"]), _traj("p1", [])]
    # eval_fraction=0.0 -> everything is in the fit split
    examples = build_step_sft_examples(trajectories, eval_fraction=0.0, seed=0)
    assert {e.problem_id for e in examples} == {"p0"}
