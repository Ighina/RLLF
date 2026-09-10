"""Base-model adapters.

The base model `M` is never fine-tuned: it only proposes continuations.  Every
decoder in this package talks to it through :class:`BaseModelAdapter`, so the
same guided decoder runs against a Hugging Face checkpoint, a vLLM server or an
API-backed model.

Two token-level capabilities matter:

* `next_token_logits` -- required for logit-level guidance (stage 4);
* `generate_step_candidates` -- required for step-level reranking (stage 2),
  which is the recommended starting point and works with API models that never
  expose logits.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..protocol import STEP_TOKEN, build_prompt


class BaseModelAdapter:
    """Interface for a frozen language model."""

    #: Set by subclasses that can return a next-token distribution.
    supports_logits: bool = False

    @property
    def tokenizer(self) -> Any:
        raise NotImplementedError

    def next_token_logits(
        self, question: str, history: Sequence[str], current_step: str
    ) -> "Any":
        """Return a 1-D tensor of logits over the vocabulary."""
        raise NotImplementedError

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Free-form completion of a raw prompt."""
        raise NotImplementedError

    def generate_step_candidates(
        self,
        question: str,
        history: Sequence[str],
        num_candidates: int = 4,
        **kwargs: Any,
    ) -> List[str]:
        """Sample `num_candidates` complete next steps, `<step>` stripped."""
        raise NotImplementedError


# --------------------------------------------------------------------------
# Hugging Face
# --------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You solve maths problems one short reasoning step at a time.\n"
    f"Rules: write ONE step, then the marker {STEP_TOKEN}, then stop. "
    "Keep each step to a single sentence that states its arithmetic explicitly "
    "(for example `7 * 6 = 42`). "
    "When the problem is solved, write `Final answer: <answer>` instead of a step."
)

#: A worked example in the target format.  An instruction alone does not get an
#: un-finetuned model to emit `<step>`; a demonstration does.
DEFAULT_FEW_SHOT: Tuple[Tuple[str, str], ...] = (
    (
        "A box holds 6 pens. A shop has 7 such boxes and gives away 5 pens. How many pens are left?",
        f"There are 7 boxes with 6 pens each, so 7 * 6 = 42. {STEP_TOKEN}\n"
        f"After giving away 5 pens, 42 - 5 = 37. {STEP_TOKEN}\n"
        "Final answer: 37",
    ),
)

#: Boundaries that end a step.  `<step>` is the protocol; the others are the
#: shapes a model that has not been fine-tuned on the protocol falls back to,
#: and the adapter normalises them all to one step (see `truncate_at_boundary`).
DEFAULT_STOP_STRINGS: Tuple[str, ...] = (STEP_TOKEN, "\n\n")


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0
    seed: Optional[int] = None


class HFBaseModel(BaseModelAdapter):
    """Frozen `transformers` causal LM.

    The model is put in eval mode with gradients disabled and is never touched
    by training: the policy is the only trainable component in the system.
    """

    supports_logits = True

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Any = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        generation: Optional[GenerationConfig] = None,
        model: Any = None,
        tokenizer: Any = None,
        use_chat_template: bool = True,
        few_shot: Optional[Sequence[Tuple[str, str]]] = None,
        stop_strings: Sequence[str] = DEFAULT_STOP_STRINGS,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype
        )
        self.device = resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.system_prompt = system_prompt
        self.few_shot = list(DEFAULT_FEW_SHOT if few_shot is None else few_shot)
        self.stop_strings = list(stop_strings)
        self.generation = generation or GenerationConfig()
        self.use_chat_template = use_chat_template and getattr(
            self._tokenizer, "chat_template", None
        )
        self._cache: Dict[str, Any] = {}

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    # -- prompting --------------------------------------------------------
    def render_prompt(self, question: str, history: Sequence[str], current_step: str = "") -> str:
        continuation = "\n".join(f"{s.strip()} {STEP_TOKEN}" for s in history if s.strip())
        if continuation:
            continuation += "\n"
        if self.use_chat_template:
            messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
            for shot_question, shot_answer in self.few_shot:
                messages.append({"role": "user", "content": f"Question: {shot_question}"})
                messages.append({"role": "assistant", "content": shot_answer})
            messages.append({"role": "user", "content": f"Question: {question}"})
            prefix = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return f"{prefix}{continuation}{current_step}"
        shots = "".join(
            f"Question: {q}\n{a}\n\n" for q, a in self.few_shot
        )
        body = build_prompt(question, history, current_step)
        return f"{self.system_prompt}\n\n{shots}{body}"

    # -- token level ------------------------------------------------------
    def next_token_logits(
        self, question: str, history: Sequence[str], current_step: str
    ):
        prompt = self.render_prompt(question, history, current_step)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model(**inputs)
        return output.logits[0, -1].float().cpu()

    # -- sequence level ---------------------------------------------------
    def generate(self, prompt: str, **kwargs: Any) -> str:
        config = {**self.generation.__dict__, **kwargs}
        seed = config.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(seed)
        stop_strings = config.pop("stop_strings", self.stop_strings)
        if config.get("temperature", 1.0) == 0.0:
            config["do_sample"] = False
            config.pop("temperature")
            config.pop("top_p", None)
            config.pop("top_k", None)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        extra: Dict[str, Any] = {}
        if stop_strings:
            extra = {"stop_strings": list(stop_strings), "tokenizer": self._tokenizer}
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                pad_token_id=self._tokenizer.pad_token_id,
                **config,
                **extra,
            )
        generated = output[0, inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def generate_step_candidates(
        self,
        question: str,
        history: Sequence[str],
        num_candidates: int = 4,
        **kwargs: Any,
    ) -> List[str]:
        prompt = self.render_prompt(question, history)
        config = {**self.generation.__dict__, **kwargs}
        seed = config.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(seed)
        if num_candidates > 1:
            config["do_sample"] = True
        if not config.get("do_sample", True):
            config.pop("temperature", None)
            config.pop("top_p", None)
            config.pop("top_k", None)
        stop_strings = config.pop("stop_strings", self.stop_strings)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                num_return_sequences=num_candidates,
                pad_token_id=self._tokenizer.pad_token_id,
                stop_strings=list(stop_strings),
                tokenizer=self._tokenizer,
                **config,
            )
        prefix_len = inputs["input_ids"].shape[-1]
        texts = self._tokenizer.batch_decode(output[:, prefix_len:], skip_special_tokens=True)
        return [truncate_at_boundary(t) for t in texts]


def truncate_at_boundary(text: str, extra_boundaries: Sequence[str] = ("\n\n",)) -> str:
    """Cut a sampled continuation at its first step boundary.

    `<step>` is the protocol, but a model that was never fine-tuned on it falls
    back to a blank line or a `Step 2:` heading.  Normalising all of these to a
    single step here is what keeps the rest of the system -- dataset, verifier,
    policy -- on one boundary convention.
    """
    head = text.split(STEP_TOKEN)[0]
    for boundary in extra_boundaries:
        head = head.split(boundary)[0]
    head = re.sub(r"\n\s*(?:Step\s*\d+\s*[:.\)]).*", "", head, flags=re.DOTALL)
    head = re.sub(r"^\s*(?:Step\s*\d+\s*[:.\)])\s*", "", head)
    return head.strip()


# --------------------------------------------------------------------------
# Offline / test doubles
# --------------------------------------------------------------------------


class ScriptedBaseModel(BaseModelAdapter):
    """Replays fixed candidate steps -- used for tests and offline demos.

    `script[question]` is a list (one entry per step index) of candidate step
    strings.  The last entry may be a final answer segment.
    """

    def __init__(self, script: Dict[str, Sequence[Sequence[str]]], seed: int = 0):
        self.script = {q: [list(c) for c in steps] for q, steps in script.items()}
        self.rng = random.Random(seed)

    @property
    def tokenizer(self) -> Any:  # pragma: no cover - not tokenizer-based
        raise NotImplementedError("ScriptedBaseModel has no tokenizer")

    def _candidates(self, question: str, history: Sequence[str]) -> List[str]:
        steps = self.script.get(question, [])
        index = len(history)
        return list(steps[index]) if index < len(steps) else []

    def generate_step_candidates(
        self, question: str, history: Sequence[str], num_candidates: int = 4, **kwargs: Any
    ) -> List[str]:
        candidates = self._candidates(question, history)
        if not candidates:
            return []
        picks = [candidates[i % len(candidates)] for i in range(num_candidates)]
        self.rng.shuffle(picks)
        return picks

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return ""


class CharTokenizer:
    """Minimal character tokenizer for the toy model."""

    def __init__(self, vocab: Iterable[str]):
        chars = sorted(set(vocab))
        self.itos: List[str] = chars
        self.stoi: Dict[str, int] = {c: i for i, c in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


class ToyCharModel(BaseModelAdapter):
    """A tiny character-level model that branches over scripted continuations.

    At each position it spreads probability mass over the next characters of
    every scripted continuation still consistent with the current step, so a
    guided decoder genuinely has a choice to make.  This makes the token-level
    guidance loop testable without downloading a checkpoint.
    """

    supports_logits = True

    def __init__(
        self,
        continuations: Dict[str, Sequence[str]],
        prior: Optional[Dict[str, float]] = None,
        noise: float = 0.02,
    ):
        self.continuations = {q: list(c) for q, c in continuations.items()}
        self.prior = dict(prior or {})
        self.noise = noise
        alphabet = {ch for options in self.continuations.values() for text in options for ch in text}
        alphabet.update(STEP_TOKEN)
        self._tokenizer = CharTokenizer(alphabet)

    @property
    def tokenizer(self) -> CharTokenizer:
        return self._tokenizer

    def next_token_logits(self, question: str, history: Sequence[str], current_step: str):
        import torch

        logits = torch.full((self._tokenizer.vocab_size,), math.log(self.noise))
        options = [
            text
            for text in self.continuations.get(question, [])
            if text.startswith(current_step) and len(text) > len(current_step)
        ]
        for text in options:
            char = text[len(current_step)]
            weight = self.prior.get(text, 1.0)
            index = self._tokenizer.stoi[char]
            logits[index] = torch.logaddexp(logits[index], torch.tensor(math.log(weight)))
        return logits

    def generate_step_candidates(
        self, question: str, history: Sequence[str], num_candidates: int = 4, **kwargs: Any
    ) -> List[str]:
        options = self.continuations.get(question, [])
        return [truncate_at_boundary(options[i % len(options)]) for i in range(min(num_candidates, len(options)))]

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return ""
