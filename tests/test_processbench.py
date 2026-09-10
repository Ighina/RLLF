"""ProcessBench loading, answer recovery, and splitting.

The loader is exercised against an in-memory stand-in for the Hub dataset, so
these tests are hermetic -- no network, no cache.
"""

import pytest

from rllf.data import processbench
from rllf.data.processbench import extract_boxed, load_processbench, reference_answer


def test_extract_boxed_handles_nested_braces():
    assert extract_boxed(r"so \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{6}") == "6"
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"  # last one wins
    assert extract_boxed("no box here") is None


def test_reference_answer_prefers_boxed():
    answer, source = reference_answer(["step one", r"therefore \boxed{42}."])
    assert (answer, source) == ("42", "boxed")


def test_reference_answer_falls_back_to_the_last_number():
    answer, source = reference_answer(["...", "the red rope is 20 centimeters."])
    assert (answer, source) == ("20", "last_number")


def test_reference_answer_strips_thousands_separators():
    assert reference_answer(["he earned 1,200 dollars."])[0] == "1200"


def test_reference_answer_reports_failure():
    assert reference_answer(["no numbers at all"]) == (None, "none")


class _FakeDataset(dict):
    pass


@pytest.fixture
def fake_hub(monkeypatch):
    rows = [
        # correct, boxed -> kept
        {"id": "gsm8k-0", "generator": "m1", "problem": "Q1", "steps": ["a", r"\boxed{1}"],
         "final_answer_correct": True, "label": -1},
        # a second correct solution to the SAME problem -> deduplicated
        {"id": "gsm8k-1", "generator": "m2", "problem": "Q1", "steps": ["b", r"\boxed{1}"],
         "final_answer_correct": True, "label": -1},
        # wrong final answer -> dropped
        {"id": "gsm8k-2", "generator": "m1", "problem": "Q2", "steps": ["c", r"\boxed{9}"],
         "final_answer_correct": False, "label": 0},
        # correct but no recoverable answer -> dropped
        {"id": "gsm8k-3", "generator": "m1", "problem": "Q3", "steps": ["no numbers"],
         "final_answer_correct": True, "label": -1},
        {"id": "gsm8k-4", "generator": "m1", "problem": "Q4", "steps": [r"\boxed{4}"],
         "final_answer_correct": True, "label": -1},
        {"id": "gsm8k-5", "generator": "m1", "problem": "Q5", "steps": ["the total is 5 apples"],
         "final_answer_correct": True, "label": -1},
    ]
    monkeypatch.setattr(processbench, "load_dataset", lambda *a, **k: _FakeDataset(gsm8k=rows), raising=False)
    monkeypatch.setitem(
        __import__("sys").modules, "datasets", type("m", (), {"load_dataset": lambda *a, **k: _FakeDataset(gsm8k=rows)})
    )
    return rows


def test_loader_filters_and_deduplicates(fake_hub):
    split = load_processbench(subsets=("gsm8k",), val_fraction=0.5, seed=0)
    problems = split.validation + split.test
    questions = [p.question for p in problems]
    assert sorted(questions) == ["Q1", "Q4", "Q5"]  # Q2 wrong, Q3 unrecoverable
    assert len(questions) == len(set(questions))  # Q1 appears once


def test_loader_recovers_references(fake_hub):
    split = load_processbench(subsets=("gsm8k",), val_fraction=0.0, seed=0)
    by_question = {p.question: p for p in split.test}
    assert by_question["Q1"].reference == "1"
    assert by_question["Q5"].reference == "5"
    assert by_question["Q5"].metadata["answer_source"] == "last_number"


def test_split_is_disjoint_and_sized(fake_hub):
    split = load_processbench(subsets=("gsm8k",), val_fraction=0.34, seed=1)
    assert len(split.validation) == 1
    assert len(split.test) == 2
    assert not {p.question for p in split.validation} & {p.question for p in split.test}


def test_metadata_does_not_shadow_the_collector_generator(fake_hub):
    """ProcessBench's `generator` names the model that wrote the *reference*."""
    split = load_processbench(subsets=("gsm8k",), val_fraction=0.0, seed=0)
    metadata = split.test[0].metadata
    assert "generator" not in metadata
    assert metadata["reference_generator"] == "m1"


def test_unknown_subset_is_rejected(fake_hub):
    with pytest.raises(ValueError, match="unknown ProcessBench subset"):
        load_processbench(subsets=("nope",))
