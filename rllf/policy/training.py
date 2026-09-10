"""Training objectives for `pi_phi`.

Three objectives, in the order the methodology recommends adopting them:

1. :func:`train_supervised` -- predict the recorded Lean/outcome rewards
   (masked BCE).  Start here.
2. :func:`train_pairwise` / :func:`train_listwise` -- rank a verifiable step
   above a non-verifiable sibling.  Use when absolute labels are noisy but
   relative quality is reliable.
3. :class:`PolicyGradientTrainer` -- REINFORCE over the decoding decisions,
   with returns built from Lean step rewards and the final-answer reward.
   Most expensive and least stable; add last.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ..data.dataset import (
    IGNORE,
    PairExample,
    PairwiseDataset,
    ProcessRewardDataset,
    StepExample,
)
from ..data.trajectory import StepRecord, Trajectory
from ..evaluation.metrics import binary_metrics, pairwise_ranking_accuracy
from ..state import PolicyState
from .model import ProcessRewardModel
from .scoring import TemperatureScaler


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    #: lambda in L = BCE(step) + lambda * BCE(outcome).
    outcome_weight: float = 0.5
    #: Weight of the auxiliary ranking loss when pairs are supplied.
    rank_weight: float = 0.0
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0  # 0 disables per-step logging
    max_length: int = 512
    #: Stop after this many epochs without held-out improvement (0 disables).
    early_stopping_patience: int = 3
    #: Held-out metric used to pick the checkpoint: "<head>_<metric>".
    select_by: str = "step_auc"


def _resolve_device(config: TrainConfig) -> torch.device:
    if config.device:
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _inference_device(model: ProcessRewardModel, config: TrainConfig) -> torch.device:
    """Where to run evaluation: follow the model unless a device was requested.

    Training moves the model itself, but evaluation must never relocate the
    batch to a device the model is not on.
    """
    if config.device:
        return torch.device(config.device)
    return next(model.parameters()).device


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE over the entries that actually carry a label."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * mask).sum() / mask.sum()


def _release_cache(device: torch.device) -> None:
    """Return cached blocks between epochs; the MPS allocator hoards them."""
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.02, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------
# 1. Supervised reward prediction
# --------------------------------------------------------------------------


def train_supervised(
    model: ProcessRewardModel,
    train_examples: Sequence[StepExample],
    eval_examples: Optional[Sequence[StepExample]] = None,
    config: Optional[TrainConfig] = None,
    pairs: Optional[Sequence[PairExample]] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Fit the step and outcome heads on recorded verifier labels."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = _resolve_device(config)
    model.to(device)

    dataset = ProcessRewardDataset(train_examples, model.tokenizer, config.max_length)
    loader = dataset.loader(batch_size=config.batch_size, shuffle=True)
    pair_loader = None
    if pairs and config.rank_weight > 0:
        pair_dataset = PairwiseDataset(pairs, model.tokenizer, config.max_length)
        pair_loader = pair_dataset.loader(batch_size=max(2, config.batch_size // 2), shuffle=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = max(1, len(loader) * config.epochs)
    scheduler = _make_scheduler(optimizer, total_steps, config.warmup_ratio)

    history: Dict[str, Any] = {"train_loss": [], "eval": []}
    global_step = 0
    best_score, best_state, epochs_without_gain = -float("inf"), None, 0
    for epoch in range(config.epochs):
        model.train()
        pair_iterator = iter(pair_loader) if pair_loader else None
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["input_ids"], batch["attention_mask"])
            loss_step = masked_bce(output["step_logit"], batch["step_labels"], batch["step_mask"])
            loss_outcome = masked_bce(
                output["outcome_logit"], batch["outcome_labels"], batch["outcome_mask"]
            )
            loss = loss_step + config.outcome_weight * loss_outcome
            if pair_iterator is not None:
                try:
                    pair_batch = next(pair_iterator)
                except StopIteration:
                    pair_iterator = iter(pair_loader)
                    pair_batch = next(pair_iterator)
                loss = loss + config.rank_weight * _pair_loss(model, pair_batch, device)

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.detach())
            n_batches += 1
            global_step += 1
            if config.log_every and global_step % config.log_every == 0:
                print(f"step {global_step}/{total_steps} loss={float(loss.detach()):.4f}")

        mean_loss = epoch_loss / max(1, n_batches)
        history["train_loss"].append(mean_loss)
        _release_cache(device)
        record: Dict[str, Any] = {"epoch": epoch, "train_loss": mean_loss}
        if eval_examples:
            record.update(evaluate_reward_model(model, eval_examples, config))
            history["eval"].append(record)
        if on_epoch_end:
            on_epoch_end(epoch, record)

        if eval_examples:
            # Keep the best held-out checkpoint: a reward model that has
            # memorised its training steps guides decoding worse than one that
            # stopped earlier, and its probabilities are badly calibrated.
            score = _selection_score(record, config.select_by)
            if score > best_score:
                best_score = score
                # Snapshot to CPU: an accelerator-resident copy of every
                # parameter, held across epochs, competes with training itself
                # for device memory (and MPS does not give it back readily).
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
                epochs_without_gain = 0
            else:
                epochs_without_gain += 1
                if config.early_stopping_patience and epochs_without_gain >= config.early_stopping_patience:
                    history["stopped_early_at_epoch"] = epoch
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        history["best_score"] = best_score
        history["select_by"] = config.select_by
    return history


def _selection_score(record: Dict[str, Any], select_by: str) -> float:
    head, _, metric = select_by.partition("_")
    values = record.get(head)
    if isinstance(values, dict) and metric in values:
        return float(values[metric])
    return -float(record.get("train_loss", 0.0))


def _pair_loss(model: ProcessRewardModel, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """-log sigmoid(s(x+) - s(x-)) over a batch of preference pairs."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    n_pairs = int(batch["num_pairs"])
    logits = model(input_ids, attention_mask)["step_logit"]
    preferred, rejected = logits[:n_pairs], logits[n_pairs:]
    return -F.logsigmoid(preferred - rejected).mean()


@torch.no_grad()
def predict(
    model: ProcessRewardModel,
    examples: Sequence[StepExample],
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Step / outcome probabilities and logits for a set of states."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    out: Dict[str, List[float]] = {"step_prob": [], "outcome_prob": [], "step_logit": [], "outcome_logit": []}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = model.tokenizer(
            [e.text for e in chunk],
            padding=True,
            truncation=True,
            max_length=model.config.max_length,
            return_tensors="pt",
        )
        result = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        out["step_logit"].extend(result["step_logit"].flatten().cpu().tolist())
        out["outcome_logit"].extend(result["outcome_logit"].flatten().cpu().tolist())
        out["step_prob"].extend(torch.sigmoid(result["step_logit"]).flatten().cpu().tolist())
        out["outcome_prob"].extend(torch.sigmoid(result["outcome_logit"]).flatten().cpu().tolist())
    if was_training:
        model.train()
    return out


def evaluate_reward_model(
    model: ProcessRewardModel,
    examples: Sequence[StepExample],
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Held-out quality of the two reward heads, including calibration."""
    config = config or TrainConfig()
    predictions = predict(model, examples, config.batch_size, _inference_device(model, config))
    step_probs, step_labels = [], []
    outcome_probs, outcome_labels = [], []
    for i, example in enumerate(examples):
        if example.step_reward != IGNORE:
            step_probs.append(predictions["step_prob"][i])
            step_labels.append(example.step_reward)
        if example.outcome_reward != IGNORE:
            outcome_probs.append(predictions["outcome_prob"][i])
            outcome_labels.append(example.outcome_reward)
    return {
        "step": binary_metrics(step_probs, step_labels),
        "outcome": binary_metrics(outcome_probs, outcome_labels),
    }


def fit_calibrators(
    model: ProcessRewardModel,
    examples: Sequence[StepExample],
    config: Optional[TrainConfig] = None,
) -> Tuple[TemperatureScaler, TemperatureScaler]:
    """Fit Platt scalers on held-out states (never on the training split)."""
    config = config or TrainConfig()
    predictions = predict(model, examples, config.batch_size, _inference_device(model, config))
    step_logits, step_labels, outcome_logits, outcome_labels = [], [], [], []
    for i, example in enumerate(examples):
        if example.step_reward != IGNORE:
            step_logits.append(predictions["step_logit"][i])
            step_labels.append(example.step_reward)
        if example.outcome_reward != IGNORE:
            outcome_logits.append(predictions["outcome_logit"][i])
            outcome_labels.append(example.outcome_reward)
    step_scaler = TemperatureScaler().fit(torch.tensor(step_logits), torch.tensor(step_labels))
    outcome_scaler = TemperatureScaler().fit(
        torch.tensor(outcome_logits), torch.tensor(outcome_labels)
    )
    return step_scaler, outcome_scaler


# --------------------------------------------------------------------------
# 2. Ranking objectives
# --------------------------------------------------------------------------


def train_pairwise(
    model: ProcessRewardModel,
    pairs: Sequence[PairExample],
    eval_pairs: Optional[Sequence[PairExample]] = None,
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """L_rank = -log sigmoid(s(x+) - s(x-))."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = _resolve_device(config)
    model.to(device)
    dataset = PairwiseDataset(pairs, model.tokenizer, config.max_length)
    loader = dataset.loader(batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = _make_scheduler(optimizer, max(1, len(loader) * config.epochs), config.warmup_ratio)

    history: Dict[str, Any] = {"train_loss": [], "eval": []}
    for _ in range(config.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            loss = _pair_loss(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            total += float(loss.detach())
            count += 1
        history["train_loss"].append(total / max(1, count))
        if eval_pairs:
            history["eval"].append(evaluate_pairs(model, eval_pairs, config))
    return history


@torch.no_grad()
def evaluate_pairs(
    model: ProcessRewardModel, pairs: Sequence[PairExample], config: Optional[TrainConfig] = None
) -> Dict[str, float]:
    config = config or TrainConfig()
    device = _inference_device(model, config)
    model.eval()
    preferred, rejected = [], []
    for start in range(0, len(pairs), config.batch_size):
        chunk = pairs[start : start + config.batch_size]
        texts = [p.preferred_text for p in chunk] + [p.rejected_text for p in chunk]
        batch = model.tokenizer(
            texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt"
        )
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))["step_logit"]
        preferred.extend(logits[: len(chunk)].cpu().tolist())
        rejected.extend(logits[len(chunk) :].cpu().tolist())
    return {
        "n_pairs": float(len(pairs)),
        "ranking_accuracy": pairwise_ranking_accuracy(preferred, rejected),
        "margin": (sum(preferred) - sum(rejected)) / max(1, len(pairs)),
    }


def train_listwise(
    model: ProcessRewardModel,
    groups: Sequence[Sequence[StepExample]],
    config: Optional[TrainConfig] = None,
) -> Dict[str, Any]:
    """Listwise ranking: softmax cross-entropy over the verified members of a group."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = _resolve_device(config)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    order = list(range(len(groups)))
    rng = random.Random(config.seed)
    history: Dict[str, Any] = {"train_loss": []}
    for _ in range(config.epochs):
        model.train()
        rng.shuffle(order)
        total, count = 0.0, 0
        for index in order:
            group = list(groups[index])
            targets = torch.tensor(
                [1.0 if e.step_reward > 0.5 else 0.0 for e in group], device=device
            )
            if targets.sum() == 0:
                continue
            batch = model.tokenizer(
                [e.text for e in group],
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))["step_logit"]
            # Cross-entropy against the uniform distribution over verified members.
            loss = -(torch.log_softmax(logits, dim=-1) * (targets / targets.sum())).sum()
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            count += 1
        history["train_loss"].append(total / max(1, count))
    return history


# --------------------------------------------------------------------------
# 3. Policy gradient
# --------------------------------------------------------------------------


@dataclass
class PolicyGradientConfig:
    epochs: int = 1
    num_candidates: int = 4
    max_steps: int = 6
    gamma: float = 0.95
    outcome_weight: float = 1.0
    entropy_weight: float = 0.01
    learning_rate: float = 1e-4
    baseline_momentum: float = 0.9
    normalize_advantages: bool = True
    grad_clip: float = 1.0
    temperature: float = 1.0
    device: Optional[str] = None
    seed: int = 0


@dataclass
class Rollout:
    trajectory: Trajectory
    log_probs: List[torch.Tensor] = field(default_factory=list)
    entropies: List[torch.Tensor] = field(default_factory=list)
    step_rewards: List[float] = field(default_factory=list)
    outcome_reward: float = 0.0


class PolicyGradientTrainer:
    """REINFORCE over step selection.

    The action at time `t` is *which* of the base model's candidate steps to
    accept, so the policy is trained on exactly the decision it makes at
    inference time.  Rewards come from Lean (per step) and the outcome checker
    (at termination):

        G_t = sum_{k>=t} gamma^{k-t} r_step^k + lambda_outcome * r_outcome
    """

    def __init__(
        self,
        model: ProcessRewardModel,
        base_model: Any,
        verifier: Any,
        outcome_checker: Any,
        config: Optional[PolicyGradientConfig] = None,
    ):
        self.model = model
        self.base_model = base_model
        self.verifier = verifier
        self.outcome_checker = outcome_checker
        self.config = config or PolicyGradientConfig()
        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.baseline = 0.0
        self.rng = random.Random(self.config.seed)

    # -- scoring with gradients ------------------------------------------
    def _candidate_scores(self, state: PolicyState, candidates: Sequence[str]) -> torch.Tensor:
        texts = [state.with_step(c).render() for c in candidates]
        batch = self.model.tokenizer(
            texts, padding=True, truncation=True, max_length=self.model.config.max_length, return_tensors="pt"
        )
        output = self.model(
            batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)
        )
        return output["step_logit"] + self.config.outcome_weight * output["outcome_logit"]

    def rollout(self, question: str, reference: Any) -> Rollout:
        """Sample one trajectory by choosing among base-model candidates."""
        history: List[str] = []
        records: List[StepRecord] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        step_rewards: List[float] = []
        final_answer = ""

        for _ in range(self.config.max_steps):
            state = PolicyState(question, list(history))
            candidates = list(
                self.base_model.generate_step_candidates(
                    question, history, num_candidates=self.config.num_candidates
                )
            )
            if not candidates:
                break
            scores = self._candidate_scores(state, candidates) / self.config.temperature
            distribution = torch.distributions.Categorical(logits=scores)
            action = distribution.sample()
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            chosen = candidates[int(action)]

            from ..protocol import contains_final_answer, extract_final_answer

            if contains_final_answer(chosen):
                final_answer = extract_final_answer(chosen)
                break
            result = self.verifier.verify(question, history, chosen)
            step_rewards.append(result.reward)
            records.append(
                StepRecord(
                    text=chosen,
                    formalization=result.formalization,
                    verified=result.verified,
                    reward=result.reward,
                    status=result.status,
                    error=result.error,
                    diagnostics=result.diagnostics,
                )
            )
            history.append(chosen)

        outcome = self.outcome_checker.check(question, final_answer, reference)
        trajectory = Trajectory(
            question=question,
            steps=records,
            final_answer=final_answer,
            outcome_reward=outcome.reward,
            reference=str(reference),
            metadata={"source": "policy_gradient_rollout"},
        )
        return Rollout(trajectory, log_probs, entropies, step_rewards, outcome.reward)

    def returns(self, rollout: Rollout) -> List[float]:
        """Discounted step returns plus the (undiscounted) outcome bonus."""
        gamma = self.config.gamma
        n_actions = len(rollout.log_probs)
        rewards = list(rollout.step_rewards) + [0.0] * (n_actions - len(rollout.step_rewards))
        returns: List[float] = [0.0] * n_actions
        running = 0.0
        for t in range(n_actions - 1, -1, -1):
            running = rewards[t] + gamma * running
            returns[t] = running + self.config.outcome_weight * rollout.outcome_reward
        return returns

    def step(self, batch: Sequence[Tuple[str, Any]]) -> Dict[str, float]:
        """One REINFORCE update over a batch of `(question, reference)` items."""
        self.model.train()
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        all_returns: List[float] = []
        rollouts: List[Rollout] = []
        for question, reference in batch:
            rollout = self.rollout(question, reference)
            if not rollout.log_probs:
                continue
            rollouts.append(rollout)
            log_probs.extend(rollout.log_probs)
            entropies.extend(rollout.entropies)
            all_returns.extend(self.returns(rollout))
        if not log_probs:
            return {"loss": 0.0, "mean_return": 0.0, "n_actions": 0.0}

        returns = torch.tensor(all_returns, dtype=torch.float, device=self.device)
        mean_return = float(returns.mean())
        self.baseline = (
            self.config.baseline_momentum * self.baseline
            + (1 - self.config.baseline_momentum) * mean_return
        )
        advantages = returns - self.baseline
        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -(torch.stack(log_probs) * advantages).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss - self.config.entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "entropy": float(entropy.detach()),
            "mean_return": mean_return,
            "baseline": self.baseline,
            "n_actions": float(len(log_probs)),
            "outcome_accuracy": sum(r.outcome_reward for r in rollouts) / max(1, len(rollouts)),
        }

    def train(
        self, items: Sequence[Tuple[str, Any]], batch_size: int = 4
    ) -> List[Dict[str, float]]:
        order = list(items)
        history: List[Dict[str, float]] = []
        for _ in range(self.config.epochs):
            self.rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                history.append(self.step(order[start : start + batch_size]))
        return history
