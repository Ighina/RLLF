"""Data-generation pipeline: questions -> verified trajectories -> dataset D.

    D = {(q, tau, r_step^{1:T}, r_outcome)}

Trajectories may come from one base model or several, at different
temperatures, prompting styles and seeds.  Diversity is the point: the policy
has to see a range of reasoning styles and failure modes, and a dataset drawn
from a single decoding configuration teaches it to score that configuration
rather than to score reasoning.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .data.trajectory import Trajectory, save_trajectories
from .generation.guided_decoder import (
    MODE_PLAIN,
    VERIFY_NONE,
    DecodeConfig,
    GuidedDecoder,
)
from .protocol import extract_final_answer
from .verification.base import OutcomeChecker, StepVerifier


@dataclass
class Problem:
    """A question with a trusted reference answer."""

    question: str
    reference: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Problem":
        known = {"question", "reference", "metadata"}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(
            question=payload["question"],
            reference=payload.get("reference", payload.get("answer")),
            metadata={**payload.get("metadata", {}), **extra},
        )


def load_problems(path: str | Path) -> List[Problem]:
    """Read a JSONL file of `{"question": ..., "reference": ...}` records."""
    problems: List[Problem] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                problems.append(Problem.from_dict(json.loads(line)))
    return problems


@dataclass
class GeneratorSpec:
    """One trajectory source: a model plus its decoding configuration."""

    name: str
    model: Any
    num_samples: int = 4
    temperature: float = 0.8
    top_p: float = 0.95
    max_steps: int = 8
    seed: int = 0
    #: Extra kwargs forwarded to the model's candidate sampler (e.g. a
    #: different system prompt, to vary the prompting strategy).
    generate_kwargs: Dict[str, Any] = field(default_factory=dict)


class TrajectoryCollector:
    """Samples trajectories, then labels every step and final answer."""

    def __init__(
        self,
        generators: Sequence[GeneratorSpec],
        verifier: Optional[StepVerifier] = None,
        outcome_checker: Optional[OutcomeChecker] = None,
        max_workers: int = 1,
    ):
        self.generators = list(generators)
        self.verifier = verifier
        self.outcome_checker = outcome_checker
        self.max_workers = max_workers

    # -- generation -------------------------------------------------------
    def sample(self, problem: Problem) -> List[Trajectory]:
        """Unlabelled trajectories for one problem, from every generator."""
        trajectories: List[Trajectory] = []
        for spec in self.generators:
            for index in range(spec.num_samples):
                config = DecodeConfig(
                    mode=MODE_PLAIN,
                    verification=VERIFY_NONE,
                    max_steps=spec.max_steps,
                    temperature=spec.temperature,
                    greedy=False,
                    seed=spec.seed + index,
                )
                decoder = GuidedDecoder(spec.model, config=config)
                result = decoder.generate(problem.question, reference=None)
                trajectory = result.trajectory
                trajectory.reference = str(problem.reference)
                trajectory.metadata.update(
                    {
                        "generator": spec.name,
                        "temperature": spec.temperature,
                        "sample_index": index,
                        "seed": spec.seed + index,
                        **problem.metadata,
                    }
                )
                trajectories.append(trajectory)
        return trajectories

    # -- labelling --------------------------------------------------------
    def annotate(self, trajectory: Trajectory, reference: Any = None) -> Trajectory:
        """Attach `r_step` to every step and `r_outcome` to the trajectory."""
        reference = reference if reference is not None else trajectory.reference
        if self.verifier is not None:
            history: List[str] = []
            for record in trajectory.steps:
                result = self.verifier.verify(trajectory.question, history, record.text)
                record.formalization = result.formalization
                record.verified = result.verified
                record.reward = result.reward
                record.status = result.status
                record.error = result.error
                record.diagnostics = dict(result.diagnostics)
                history.append(record.text)
        if self.outcome_checker is not None and reference is not None:
            outcome = self.outcome_checker.check(
                trajectory.question, trajectory.final_answer, reference
            )
            trajectory.outcome_reward = outcome.reward
            trajectory.metadata["outcome"] = outcome.to_dict()
            trajectory.metadata["extracted_answer"] = extract_final_answer(trajectory.final_answer)
        return trajectory

    def collect(
        self,
        problems: Sequence[Problem],
        output_path: Optional[str | Path] = None,
        progress: bool = False,
    ) -> List[Trajectory]:
        """Full pipeline: sample, verify, check, and optionally persist."""
        trajectories: List[Trajectory] = []
        for index, problem in enumerate(problems):
            if progress:
                print(f"[{index + 1}/{len(problems)}] {problem.question[:70]}")
            sampled = self.sample(problem)
            trajectories.extend(self._annotate_all(sampled, problem.reference))
        if output_path:
            save_trajectories(output_path, trajectories)
        return trajectories

    def _annotate_all(self, trajectories: Sequence[Trajectory], reference: Any) -> List[Trajectory]:
        if self.max_workers > 1 and len(trajectories) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                return list(pool.map(lambda t: self.annotate(t, reference), trajectories))
        return [self.annotate(t, reference) for t in trajectories]


def annotate_trajectories(
    trajectories: Iterable[Trajectory],
    verifier: Optional[StepVerifier] = None,
    outcome_checker: Optional[OutcomeChecker] = None,
    max_workers: int = 1,
) -> List[Trajectory]:
    """Label trajectories that were generated elsewhere (e.g. loaded from disk)."""
    collector = TrajectoryCollector([], verifier, outcome_checker, max_workers)
    items = list(trajectories)
    return collector._annotate_all(items, None)


def dataset_summary(trajectories: Sequence[Trajectory]) -> Dict[str, Any]:
    """Label balance and failure-mode breakdown of a collected dataset."""
    from collections import Counter

    steps = [s for t in trajectories for s in t.steps]
    statuses = Counter(s.status for s in steps)
    categories = Counter(s.diagnostics.get("category", "unknown") for s in steps)
    return {
        "n_trajectories": len(trajectories),
        "n_questions": len({t.question for t in trajectories}),
        "n_steps": len(steps),
        "step_positive_rate": (sum(s.reward for s in steps) / len(steps)) if steps else 0.0,
        "outcome_positive_rate": (
            sum(t.outcome_reward for t in trajectories) / len(trajectories) if trajectories else 0.0
        ),
        "status_counts": dict(statuses),
        "diagnostic_counts": dict(categories),
        "generators": dict(Counter(t.metadata.get("generator", "unknown") for t in trajectories)),
    }
