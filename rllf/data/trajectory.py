"""Trajectory representation: D = {(q, tau, r_step^{1:T}, r_outcome)}."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from ..protocol import format_trajectory, split_steps


@dataclass
class StepRecord:
    """One reasoning step `s_t` together with its verification outcome."""

    text: str
    formalization: Optional[str] = None
    verified: bool = False
    reward: float = 0.0
    status: str = "unverified"
    error: Optional[str] = None
    #: Richer verifier diagnostics (parser/type-check/timeout/...), kept as
    #: auxiliary training signal even though `reward` stays binary.
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "StepRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class Trajectory:
    """A full solution attempt for one question."""

    question: str
    steps: List[StepRecord]
    final_answer: str
    outcome_reward: float = 0.0
    reference: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- derived views ----------------------------------------------------
    @property
    def step_texts(self) -> List[str]:
        return [s.text for s in self.steps]

    @property
    def step_rewards(self) -> List[float]:
        return [s.reward for s in self.steps]

    @property
    def fully_verified(self) -> bool:
        return bool(self.steps) and all(s.verified for s in self.steps)

    def history_at(self, index: int) -> List[str]:
        """Previous steps `H_t` seen before step `index`."""
        return self.step_texts[:index]

    def render(self) -> str:
        return format_trajectory(self.step_texts, self.final_answer)

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["step_rewards"] = self.step_rewards
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Trajectory":
        raw_steps = payload.get("steps", [])
        rewards = payload.get("step_rewards") or []
        steps: List[StepRecord] = []
        for i, step in enumerate(raw_steps):
            if isinstance(step, str):
                reward = float(rewards[i]) if i < len(rewards) else 0.0
                steps.append(
                    StepRecord(text=step, verified=bool(reward), reward=reward)
                )
            else:
                steps.append(StepRecord.from_dict(step))
        return cls(
            question=payload["question"],
            steps=steps,
            final_answer=payload.get("final_answer", ""),
            outcome_reward=float(payload.get("outcome_reward", 0.0)),
            reference=payload.get("reference"),
            metadata=payload.get("metadata", {}) or {},
        )

    @classmethod
    def from_text(
        cls,
        question: str,
        text: str,
        reference: Optional[str] = None,
        **metadata: Any,
    ) -> "Trajectory":
        """Build an unverified trajectory from raw `<step>`-delimited text."""
        steps, final = split_steps(text)
        return cls(
            question=question,
            steps=[StepRecord(text=s) for s in steps],
            final_answer=final,
            reference=reference,
            metadata=dict(metadata),
        )


def save_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for traj in trajectories:
            handle.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_trajectories(path: str | Path) -> List[Trajectory]:
    return list(iter_trajectories(path))


def iter_trajectories(path: str | Path) -> Iterator[Trajectory]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Trajectory.from_dict(json.loads(line))


def split_by_question(
    trajectories: Sequence[Trajectory], eval_fraction: float = 0.2, seed: int = 0
) -> tuple[List[Trajectory], List[Trajectory]]:
    """Held-out split that keeps every trajectory of a question on one side.

    Splitting on trajectories rather than questions would leak: the same
    question appears in many trajectories, so the reward model could memorise
    per-question answers instead of judging steps.
    """
    import random

    questions = sorted({t.question for t in trajectories})
    rng = random.Random(seed)
    rng.shuffle(questions)
    n_eval = int(round(len(questions) * eval_fraction))
    eval_questions = set(questions[:n_eval])
    train = [t for t in trajectories if t.question not in eval_questions]
    held_out = [t for t in trajectories if t.question in eval_questions]
    return train, held_out
