"""Encoders for the process reward model.

Two backends share one interface (an object exposing `config.hidden_size` and
returning `.last_hidden_state`), so the policy code is identical whether it
sits on a pretrained Hugging Face encoder or on the small from-scratch
transformer used for offline runs and tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

PAD_ID = 0
CLS_ID = 1
UNK_OFFSET = 2  # ids 0/1 are reserved


@dataclass
class EncoderOutput:
    last_hidden_state: torch.Tensor


class HashingTokenizer:
    """Word-hashing tokenizer with a Hugging Face-compatible call signature.

    Deterministic across processes (`hashlib`, not Python's randomised `hash`),
    so a cached dataset stays valid between runs.
    """

    def __init__(self, vocab_size: int = 32768, model_max_length: int = 512):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        self.pad_token_id = PAD_ID
        self.cls_token_id = CLS_ID

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (self.vocab_size - UNK_OFFSET) + UNK_OFFSET

    def tokenize(self, text: str) -> List[str]:
        import re

        return re.findall(r"<step>|[A-Za-z_]+|\d|[^\sA-Za-z0-9_]", text.lower())

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        limit = max_length or self.model_max_length
        ids = [CLS_ID] + [self._bucket(t) for t in self.tokenize(text)]
        # Keep the tail: the current step (the thing being scored) sits at the end.
        return ids[:1] + ids[1:][-(limit - 1) :]

    def __call__(
        self,
        texts: str | Sequence[str],
        padding: Any = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        limit = max_length or self.model_max_length
        encoded = [self.encode(t, limit) for t in texts]
        width = max(len(e) for e in encoded) if encoded else 1
        input_ids = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class HashingEncoder(nn.Module):
    """Small transformer encoder trained from scratch over hashed tokens."""

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = _EncoderConfig(hidden_size=hidden_size, max_length=max_length)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_ID)
        self.position = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> EncoderOutput:
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device).clamp_(
            max=self.config.max_length - 1
        )
        hidden = self.embedding(input_ids) + self.position(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
        return EncoderOutput(self.norm(hidden))


@dataclass
class _EncoderConfig:
    hidden_size: int
    max_length: int = 512


def build_encoder(name: Optional[str] = None, **kwargs: Any):
    """Build `(encoder, tokenizer)`.

    `name=None` (or `"hashing"`) gives the dependency-free encoder; anything
    else is loaded with `transformers.AutoModel`.
    """
    if name in (None, "", "hashing"):
        tokenizer = HashingTokenizer(
            vocab_size=kwargs.get("vocab_size", 32768),
            model_max_length=kwargs.get("max_length", 512),
        )
        return HashingEncoder(**kwargs), tokenizer
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    # A policy state is rendered as question + history + CURRENT STEP, with the
    # step being scored at the end.  Hugging Face tokenizers truncate from the
    # right by default, which would drop exactly that step and score the
    # context instead; trim the older context instead, as HashingTokenizer does.
    tokenizer.truncation_side = "left"
    encoder = AutoModel.from_pretrained(name)
    return encoder, tokenizer
