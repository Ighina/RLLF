import shutil

import pytest

from rllf.data.synthetic import SyntheticBaseModel, make_problems
from rllf.verification.lean_verifier import MockStepVerifier
from rllf.verification.outcome_checker import default_outcome_checker

lean_available = shutil.which("lean") is not None
requires_lean = pytest.mark.skipif(not lean_available, reason="lean toolchain not installed")


@pytest.fixture
def problems():
    return make_problems(6, seed=3)


@pytest.fixture
def base_model(problems):
    return SyntheticBaseModel(problems, error_rate=0.5, seed=1)


@pytest.fixture
def verifier():
    return MockStepVerifier()


@pytest.fixture
def outcome_checker():
    return default_outcome_checker()
