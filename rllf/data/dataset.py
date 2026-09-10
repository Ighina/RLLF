"""Turning verified trajectories into supervised / ranking training data."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from ..state import render
from .trajectory import Trajectory

#: Label value meaning "this example carries no supervision for that head".
IGNORE = -1.0


@dataclass
class StepExample:
    """One `(q, H_t, s_t)` state with its recorded rewards."""

    question: str
    history: List[str]
    step: str
    step_reward: float = IGNORE
    outcome_reward: float = IGNORE
    step_index: int = 0
    is_final: bool = False
    trajectory_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        if self.is_final:
            return render(self.question, self.history, final_answer=self.step)
        return render(self.question, self.history, current_step=self.step)

    @property
    def group_key(self) -> Tuple[str, Tuple[str, ...]]:
        """States that share a key are alternative continuations of one prefix."""
        return (self.question, tuple(self.history))


def build_step_examples(
    trajectories: Sequence[Trajectory],
    include_final: bool = True,
    propagate_outcome: bool = True,
) -> List[StepExample]:
    """Explode trajectories into per-step training states.

    Every intermediate step carries its Lean reward.  The trajectory's outcome
    reward is attached to the terminal state, and -- when `propagate_outcome`
    is set -- also to intermediate states, which is what turns the outcome head
    into the value model `V_phi(q, H_t, u_t)` that beam and tree search need.
    """
    examples: List[StepExample] = []
    for traj_id, traj in enumerate(trajectories):
        for index, step in enumerate(traj.steps):
            examples.append(
                StepExample(
                    question=traj.question,
                    history=traj.history_at(index),
                    step=step.text,
                    step_reward=float(step.reward),
                    outcome_reward=float(traj.outcome_reward) if propagate_outcome else IGNORE,
                    step_index=index,
                    trajectory_id=traj_id,
                    metadata={"status": step.status, **step.diagnostics},
                )
            )
        if include_final and traj.final_answer:
            examples.append(
                StepExample(
                    question=traj.question,
                    history=traj.step_texts,
                    step=traj.final_answer,
                    step_reward=IGNORE,  # the final answer is not a Lean step
                    outcome_reward=float(traj.outcome_reward),
                    step_index=len(traj.steps),
                    is_final=True,
                    trajectory_id=traj_id,
                    metadata={"terminal": True},
                )
            )
    return examples


class ProcessRewardDataset(Dataset):
    """Tokenised `StepExample`s for supervised reward prediction."""

    def __init__(
        self,
        examples: Sequence[StepExample],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> StepExample:
        return self.examples[index]

    def collate(self, batch: Sequence[StepExample]) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [example.text for example in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        step = torch.tensor([e.step_reward for e in batch], dtype=torch.float)
        outcome = torch.tensor([e.outcome_reward for e in batch], dtype=torch.float)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "step_labels": step.clamp(min=0.0),
            "step_mask": (step != IGNORE).float(),
            "outcome_labels": outcome.clamp(min=0.0),
            "outcome_mask": (outcome != IGNORE).float(),
        }

    def loader(self, batch_size: int = 16, shuffle: bool = True, **kwargs: Any):
        from torch.utils.data import DataLoader

        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, **kwargs
        )


@dataclass
class PairExample:
    """A preferred / rejected pair sharing the same generation state."""

    question: str
    history: List[str]
    preferred: str
    rejected: str
    #: "prefix" pairs share the exact step history; "question" pairs only the
    #: question, so their relative quality is a weaker signal.
    level: str = "prefix"

    @property
    def preferred_text(self) -> str:
        return render(self.question, self.history, current_step=self.preferred)

    @property
    def rejected_text(self) -> str:
        return render(self.question, self.history, current_step=self.rejected)


def build_pairs(
    examples: Sequence[StepExample],
    max_pairs_per_group: int = 8,
    allow_question_level: bool = True,
    seed: int = 0,
) -> List[PairExample]:
    """Pair a Lean-verifiable step against a non-verifiable sibling.

    Pairs are formed within the same prefix wherever possible: two steps that
    continue the *same* state differ only in quality, which is exactly the
    comparison the ranking loss should learn.
    """
    rng = random.Random(seed)
    by_prefix: Dict[Tuple[str, Tuple[str, ...]], List[StepExample]] = defaultdict(list)
    by_question: Dict[str, List[StepExample]] = defaultdict(list)
    for example in examples:
        if example.is_final or example.step_reward == IGNORE:
            continue
        by_prefix[example.group_key].append(example)
        by_question[example.question].append(example)

    pairs: List[PairExample] = []
    seen: set = set()
    covered_questions = set()

    def add(pair: PairExample) -> None:
        key = (pair.question, tuple(pair.history), pair.preferred, pair.rejected)
        if key not in seen:
            seen.add(key)
            pairs.append(pair)
    for (question, history), group in by_prefix.items():
        positives = [e for e in group if e.step_reward > 0.5]
        negatives = [e for e in group if e.step_reward <= 0.5]
        if not positives or not negatives:
            continue
        covered_questions.add(question)
        combos = [(p, n) for p in positives for n in negatives]
        rng.shuffle(combos)
        for pos, neg in combos[:max_pairs_per_group]:
            add(PairExample(question, list(history), pos.step, neg.step, "prefix"))

    if allow_question_level:
        for question, group in by_question.items():
            if question in covered_questions:
                continue
            positives = [e for e in group if e.step_reward > 0.5]
            negatives = [e for e in group if e.step_reward <= 0.5]
            if not positives or not negatives:
                continue
            combos = [(p, n) for p in positives for n in negatives]
            rng.shuffle(combos)
            for pos, neg in combos[:max_pairs_per_group]:
                # Each side keeps its own history; the shared context is the question.
                add(PairExample(question, list(pos.history), pos.step, neg.step, "question"))
    return pairs


class PairwiseDataset(Dataset):
    """Preference pairs for the ranking objective."""

    def __init__(self, pairs: Sequence[PairExample], tokenizer: Any, max_length: int = 512):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PairExample:
        return self.pairs[index]

    def collate(self, batch: Sequence[PairExample]) -> Dict[str, torch.Tensor]:
        texts = [p.preferred_text for p in batch] + [p.rejected_text for p in batch]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "num_pairs": torch.tensor(len(batch)),
        }

    def loader(self, batch_size: int = 8, shuffle: bool = True, **kwargs: Any):
        from torch.utils.data import DataLoader

        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate, **kwargs
        )


def build_listwise_groups(
    examples: Sequence[StepExample], min_size: int = 2
) -> List[List[StepExample]]:
    """Group alternative continuations of the same state for listwise ranking."""
    groups: Dict[Tuple[str, Tuple[str, ...]], List[StepExample]] = defaultdict(list)
    for example in examples:
        if not example.is_final and example.step_reward != IGNORE:
            groups[example.group_key].append(example)
    return [
        group
        for group in groups.values()
        if len(group) >= min_size and len({e.step_reward for e in group}) > 1
    ]
