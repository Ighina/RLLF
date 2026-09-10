"""The process reward model `pi_phi`.

A separate, swappable module: the base model stays frozen and this is the only
thing that trains.  It reads `question + previous steps + current step` and
predicts

* `step_logit`    -- will this step formalize and verify in Lean? (r_step)
* `outcome_logit` -- will this trajectory reach the right answer? (r_outcome)
* `action_logits` -- continue the step / close it / finish the answer (section 9)
* `token_logits`  -- optional per-token guidance scores for logit-level control

Only the first two are needed for the recommended first implementation
(supervised reward prediction + step reranking); the action and token heads are
built lazily so nothing pays for what it does not use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from .encoder import build_encoder


class Action(str, Enum):
    """Step-boundary decisions the policy can take (section 9)."""

    CONTINUE_STEP = "continue"
    CLOSE_STEP = "close_step"
    END_ANSWER = "finish"


ACTIONS: List[Action] = [Action.CONTINUE_STEP, Action.CLOSE_STEP, Action.END_ANSWER]


@dataclass
class ProcessRewardConfig:
    encoder_name: Optional[str] = None  # None -> from-scratch hashing encoder
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_length: int = 512
    dropout: float = 0.1
    pooling: str = "mean"  # "mean" | "cls" | "last"
    #: Enable the action head used for learned step-boundary decisions.
    use_action_head: bool = True
    #: Vocabulary size for the token-guidance head (stage 4).  `None` disables it.
    token_vocab_size: Optional[int] = None
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "max_length": self.max_length,
            "dropout": self.dropout,
            "pooling": self.pooling,
            "use_action_head": self.use_action_head,
            "token_vocab_size": self.token_vocab_size,
            "encoder_kwargs": dict(self.encoder_kwargs),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProcessRewardConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


class ProcessRewardModel(nn.Module):
    """Encoder + reward heads."""

    def __init__(self, config: Optional[ProcessRewardConfig] = None, encoder: Any = None, tokenizer: Any = None):
        super().__init__()
        self.config = config or ProcessRewardConfig()
        if encoder is None:
            kwargs: Dict[str, Any] = dict(self.config.encoder_kwargs)
            if self.config.encoder_name in (None, "", "hashing"):
                kwargs.setdefault("hidden_size", self.config.hidden_size)
                kwargs.setdefault("num_layers", self.config.num_layers)
                kwargs.setdefault("num_heads", self.config.num_heads)
                kwargs.setdefault("max_length", self.config.max_length)
                kwargs.setdefault("dropout", self.config.dropout)
            encoder, tokenizer = build_encoder(self.config.encoder_name, **kwargs)
        self.encoder = encoder
        self.tokenizer = tokenizer
        hidden = int(getattr(encoder.config, "hidden_size", self.config.hidden_size))
        self.hidden_size = hidden
        self.dropout = nn.Dropout(self.config.dropout)
        self.step_head = nn.Linear(hidden, 1)
        self.outcome_head = nn.Linear(hidden, 1)
        self.action_head = nn.Linear(hidden, len(ACTIONS)) if self.config.use_action_head else None
        self.token_head = (
            nn.Linear(hidden, self.config.token_vocab_size)
            if self.config.token_vocab_size
            else None
        )

    # -- forward ----------------------------------------------------------
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "cls":
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        if self.config.pooling == "last":
            index = attention_mask.sum(dim=1).clamp(min=1) - 1
            return hidden_states[torch.arange(hidden_states.size(0)), index]
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.pool(output.last_hidden_state, attention_mask))
        result = {
            "step_logit": self.step_head(pooled).squeeze(-1),
            "outcome_logit": self.outcome_head(pooled).squeeze(-1),
            "pooled": pooled,
        }
        if self.action_head is not None:
            result["action_logits"] = self.action_head(pooled)
        if return_tokens and self.token_head is not None:
            result["token_logits"] = self.token_head(pooled)
        return result

    # -- convenience ------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_texts(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}

    @torch.no_grad()
    def score_texts(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Predicted probabilities for a batch of rendered states."""
        was_training = self.training
        self.eval()
        output = self(**self.encode_texts(texts), return_tokens=True)
        if was_training:
            self.train()
        scores = {
            "step_prob": torch.sigmoid(output["step_logit"]).cpu(),
            "outcome_prob": torch.sigmoid(output["outcome_logit"]).cpu(),
            "step_logit": output["step_logit"].cpu(),
            "outcome_logit": output["outcome_logit"].cpu(),
        }
        if "action_logits" in output:
            scores["action_logits"] = output["action_logits"].cpu()
        if "token_logits" in output:
            scores["token_logits"] = output["token_logits"].cpu()
        return scores

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        import json

        (path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, map_location: Any = "cpu") -> "ProcessRewardModel":
        import json

        path = Path(path)
        config = ProcessRewardConfig.from_dict(json.loads((path / "config.json").read_text()))
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location=map_location))
        model.eval()
        return model
