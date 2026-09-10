from rllf.protocol import (
    FINAL_ANSWER_MARKER,
    STEP_TOKEN,
    build_prompt,
    contains_final_answer,
    contains_step_boundary,
    extract_final_answer,
    format_trajectory,
    split_steps,
    strip_step_token,
)


def test_format_and_split_round_trip():
    steps = ["first step", "second step"]
    text = format_trajectory(steps, "42")
    assert text.count(STEP_TOKEN) == 2
    parsed_steps, final = split_steps(text)
    assert parsed_steps == steps
    assert extract_final_answer(final) == "42"


def test_format_does_not_duplicate_the_answer_marker():
    text = format_trajectory(["a"], f"{FINAL_ANSWER_MARKER} 7")
    assert text.count(FINAL_ANSWER_MARKER) == 1


def test_split_without_boundary_returns_everything_as_final():
    steps, final = split_steps("just an answer")
    assert steps == []
    assert final == "just an answer"


def test_boundary_detection():
    assert contains_step_boundary(f"partial {STEP_TOKEN}")
    assert not contains_step_boundary("partial")
    assert contains_final_answer("Final answer: 3")
    assert contains_final_answer(r"so \boxed{3}")
    assert contains_final_answer("#### 3")
    assert not contains_final_answer("still reasoning")


def test_extract_final_answer_formats():
    assert extract_final_answer("Final answer: 12") == "12"
    assert extract_final_answer("The answer is 12") == "12"
    assert extract_final_answer(r"we get \boxed{12}") == "12"
    assert extract_final_answer("#### 12") == "12"
    assert extract_final_answer("12") == "12"


def test_strip_step_token():
    assert strip_step_token(f"  a step {STEP_TOKEN} ") == "a step"


def test_build_prompt_includes_history_and_current_step():
    prompt = build_prompt("q", ["s1", "s2"], "partial")
    assert prompt.splitlines()[0] == "Question: q"
    assert prompt.count(STEP_TOKEN) == 2
    assert prompt.endswith("partial")
