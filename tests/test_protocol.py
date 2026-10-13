from rllf.protocol import (
    FINAL_ANSWER_MARKER,
    FINAL_TOKEN,
    STEP_TOKEN,
    build_prompt,
    contains_final_answer,
    contains_step_boundary,
    extract_final_answer,
    format_trajectory,
    parse_numbered_steps,
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


def test_parse_numbered_steps_splits_a_plain_numbered_list():
    text = "1. First step here.\n2. Second step here.\n3) Third, parenthesis style."
    assert parse_numbered_steps(text) == [
        "First step here.",
        "Second step here.",
        "Third, parenthesis style.",
    ]


def test_parse_numbered_steps_folds_wrapped_lines_into_the_step_they_continue():
    # A model doesn't always keep one step to one line -- a continuation line
    # belongs to the numbered step above it, not a dropped/new entry.
    text = "1. A step that\nwraps onto a second line.\n2. A short step."
    assert parse_numbered_steps(text) == [
        "A step that wraps onto a second line.",
        "A short step.",
    ]


def test_parse_numbered_steps_ignores_preamble_before_the_first_number():
    text = "Here are the steps:\n\n1. Only step."
    assert parse_numbered_steps(text) == ["Only step."]


def test_parse_numbered_steps_empty_input():
    assert parse_numbered_steps("") == []
    assert parse_numbered_steps("no numbers here") == []


def test_final_token_is_recognised_as_a_final_answer():
    assert contains_final_answer(f"{FINAL_TOKEN} 97")
    assert not contains_final_answer("still reasoning")


def test_extract_final_answer_reads_a_plain_final_token():
    assert extract_final_answer(f"{FINAL_TOKEN} 97") == "97"


def test_extract_final_answer_prefers_boxed_over_a_final_token_wrapper():
    # `<final> \boxed{97}` -- the more specific boxed extraction still wins,
    # `<final>` doesn't need its own duplicate boxed-parsing logic.
    assert extract_final_answer(f"{FINAL_TOKEN} \\boxed{{97}}") == "97"


def test_extract_boxed_handles_nested_braces():
    from rllf.protocol import extract_boxed

    text = r"so \[\frac{1}{\tan(-60^\circ)} = \boxed{-\frac{\sqrt{3}}{3}}.\] and \boxed{2,4}"
    assert extract_boxed(text) == [r"-\frac{\sqrt{3}}{3}", "2,4"]
    assert extract_final_answer(text) == "2,4"
    assert extract_final_answer(r"answer \boxed{\frac{\sqrt{3}}{3}}") == r"\frac{\sqrt{3}}{3}"
    assert extract_boxed(r"unterminated \boxed{\frac{1}{2}") == []
    assert extract_boxed("no box") == []
