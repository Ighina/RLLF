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

from ..protocol import STEP_TOKEN, build_prompt, parse_numbered_steps


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

    def generate_full_solution(
        self, question: str, num_candidates: int = 1, **kwargs: Any
    ) -> List[str]:
        """Sample `num_candidates` complete, naturally-written solutions.

        No `<step>` protocol, no per-step early stop: the model reasons in
        free prose to a natural end (EOS or `max_new_tokens`), the way
        ProcessBench's own generator models were prompted. Used by
        `full_trajectory_rerank` to test whether the `<step>`-token
        generation-time boundary itself -- not just the base model's identity
        -- explains why a classifier trained on naturally-segmented
        FaithformBench steps transfers poorly to `rerank`'s candidates.
        """
        raise NotImplementedError

    def segment_into_steps(self, solution: str) -> List[str]:
        """Ask this same model to retroactively split a natural solution into
        atomic steps -- the ProcessBench "generate freely, then segment"
        convention, paired with `generate_full_solution`."""
        raise NotImplementedError

    def generate_step_beam(
        self,
        question: str,
        history: Sequence[str],
        num_beams: int = 4,
        num_return_sequences: int = 4,
        length_penalty: float = 1.0,
        **kwargs: Any,
    ) -> List[Any]:
        """Beam-search `num_return_sequences` continuations of the next step,
        stopped at the `<step>` boundary, each paired with the base model's
        own (`length_penalty`-normalised) cumulative log-probability for the
        tokens generated in *this* call.

        Used by `rllf.generation.beam_search.lean_constrained_beam_search`:
        real, token-level beam search for the "usual" pruning within a step,
        with the classifier intervening only at the boundary between calls.
        Returns `[(step_text, lm_score), ...]`.
        """
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

#: `generate_full_solution`'s prompt: no `<step>` instruction, no forced
#: boundary -- the ProcessBench-style "just reason step by step in prose"
#: instruction, matched to MATH-500's own `\boxed{...}` answer convention so
#: the existing `contains_final_answer`/`extract_final_answer` machinery
#: (rllf/protocol.py) reads it without any special-casing.
DEFAULT_FREE_SYSTEM_PROMPT = (
    "You solve maths problems by reasoning step by step in natural prose. "
    "Show your work, then give the final answer as \\boxed{...} on its own "
    "at the end."
)

#: `segment_into_steps`'s prompt: retroactively splits an already-written
#: solution into atomic steps, the other half of the ProcessBench convention
#: (generate freely first, segment after) -- as opposed to this codebase's
#: own generation-time `<step>` boundary.
DEFAULT_SEGMENTATION_PROMPT = (
    "Split the following worked solution into a numbered list of its atomic "
    "reasoning steps, in order. Preserve the original wording as closely as "
    "possible -- do not add, remove, or alter any mathematical content, only "
    "insert the step boundaries. Output ONLY the numbered list, one step per "
    "line, formatted exactly as `1. <step text>`.\n\nSolution:\n{solution}\n\n"
    "Numbered steps:"
)


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

    def _free_prompt(self, question: str) -> str:
        if self.use_chat_template:
            messages = [
                {"role": "system", "content": DEFAULT_FREE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"{DEFAULT_FREE_SYSTEM_PROMPT}\n\nQuestion: {question}\nSolution:"

    def generate_full_solution(
        self, question: str, num_candidates: int = 1, **kwargs: Any
    ) -> List[str]:
        prompt = self._free_prompt(question)
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
        # A natural solution needs far more room than a single `<step>`-bounded
        # step; the base `GenerationConfig.max_new_tokens` (128) is sized for
        # `generate_step_candidates`, not this.
        max_new_tokens = config.pop("max_new_tokens", 512)
        # No `stop_strings`: unlike `generate_step_candidates`, nothing should
        # cut this off early -- it must run to a natural end (EOS) so the
        # model reasons in free prose the way ProcessBench's own generators
        # were prompted, not the `<step>`-bounded protocol this codebase uses.
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                num_return_sequences=num_candidates,
                pad_token_id=self._tokenizer.pad_token_id,
                max_new_tokens=max_new_tokens,
                **config,
            )
        prefix_len = inputs["input_ids"].shape[-1]
        return self._tokenizer.batch_decode(output[:, prefix_len:], skip_special_tokens=True)

    def segment_into_steps(self, solution: str) -> List[str]:
        prompt_text = DEFAULT_SEGMENTATION_PROMPT.format(solution=solution.strip())
        if self.use_chat_template:
            prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = prompt_text
        # Greedy (no sampling, via `generate()`'s temperature=0.0 sentinel)
        # and no early-stop boundary: this is a deterministic parsing task,
        # not creative generation, and the numbered list can legitimately
        # contain blank lines the `<step>` protocol's stop strings would
        # truncate at.
        text = self.generate(prompt, temperature=0.0, max_new_tokens=1024, stop_strings=[])
        return parse_numbered_steps(text)

    def generate_step_beam(
        self,
        question: str,
        history: Sequence[str],
        num_beams: int = 4,
        num_return_sequences: int = 4,
        length_penalty: float = 1.0,
        **kwargs: Any,
    ) -> List[Any]:
        prompt = self.render_prompt(question, history)
        config = {**self.generation.__dict__, **kwargs}
        seed = config.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(seed)
        # Beam search is deterministic given the beams; sampling knobs from
        # `GenerationConfig` don't apply and HF errors if `do_sample=False`
        # (the default once `num_beams > 1`) is combined with them.
        for key in ("do_sample", "temperature", "top_p", "top_k"):
            config.pop(key, None)
        max_new_tokens = config.pop("max_new_tokens", 128)
        stop_strings = config.pop("stop_strings", self.stop_strings)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                stop_strings=list(stop_strings),
                tokenizer=self._tokenizer,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                **config,
            )
        prefix_len = inputs["input_ids"].shape[-1]
        texts = self._tokenizer.batch_decode(output.sequences[:, prefix_len:], skip_special_tokens=True)
        # `sequences_scores` is `sum(log p(token)) / length**length_penalty`
        # for each returned beam -- HF's own length normalisation, exposed
        # here as `length_penalty` rather than reimplemented (`update.md`'s
        # beta). Not populated for every generation config in every
        # `transformers` version; fall back to 0.0 (pure-classifier ranking)
        # rather than crash, but this should be verified against the
        # installed version before trusting it in a real run -- see
        # `lean-constrained-beam-search.md`'s verification checklist.
        if getattr(output, "sequences_scores", None) is not None:
            scores = output.sequences_scores.float().cpu().tolist()
        else:
            scores = [0.0] * len(texts)
        return list(zip([truncate_at_boundary(t) for t in texts], scores))


class StepNativeHFBaseModel(HFBaseModel):
    """`HFBaseModel` for a checkpoint fine-tuned by `scripts/finetune_step_native.py`.

    Two things differ from the instruction-prompted base class, both of which
    silently broke the step-native smoke test (`lean-constrained-beam-search.md`:
    0/5 problems "reached an answer"):

    * **Prompt.** The fine-tuning trained on `render_step_sft_prompt(question)`
      followed by `step <step>\\n...` lines -- raw text, no chat template, no
      few-shot demo. `render_prompt` reproduces that byte-for-byte.
    * **Decoding.** `<step>`/`<final>` were added as *special* tokens, and the
      base class decodes with `skip_special_tokens=True`, which deletes them:
      the model's `<final> 42` came back as the bare step `42`, so
      `contains_final_answer` never fired and every trajectory ran to
      `max_steps`. Here only pad/EOS are stripped; `<step>` is the stop
      boundary and `<final>` is kept in the returned text for
      `contains_final_answer`/`extract_final_answer` (`rllf/protocol.py`).

    The blank-line fallback boundary is also dropped: the model was taught the
    token, and a `\\n\\n` inside a step is content, not a boundary.
    """

    def __init__(self, model_name_or_path: str, **kwargs: Any):
        kwargs.setdefault("stop_strings", (STEP_TOKEN,))
        kwargs.setdefault("few_shot", ())
        kwargs.setdefault("use_chat_template", False)
        super().__init__(model_name_or_path, **kwargs)
        self._strip_tokens = {
            t for t in (self._tokenizer.eos_token, self._tokenizer.pad_token) if t
        }

    def render_prompt(self, question: str, history: Sequence[str], current_step: str = "") -> str:
        from ..data.step_sft import render_step_sft_prompt

        continuation = "\n".join(f"{s.strip()} {STEP_TOKEN}" for s in history if s.strip())
        if continuation:
            continuation += "\n"
        return f"{render_step_sft_prompt(question)}{continuation}{current_step}"

    def _decode(self, token_ids: Any) -> List[str]:
        texts = self._tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        cleaned = []
        for text in texts:
            for token in self._strip_tokens:
                text = text.replace(token, "")
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _finish_step(text: str) -> str:
        return text.split(STEP_TOKEN)[0].strip()

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
        return [self._finish_step(t) for t in self._decode(output[:, prefix_len:])]

    def generate_step_beam(
        self,
        question: str,
        history: Sequence[str],
        num_beams: int = 4,
        num_return_sequences: int = 4,
        length_penalty: float = 1.0,
        **kwargs: Any,
    ) -> List[Any]:
        prompt = self.render_prompt(question, history)
        config = {**self.generation.__dict__, **kwargs}
        seed = config.pop("seed", None)
        if seed is not None:
            self.torch.manual_seed(seed)
        for key in ("do_sample", "temperature", "top_p", "top_k"):
            config.pop(key, None)
        max_new_tokens = config.pop("max_new_tokens", 128)
        stop_strings = config.pop("stop_strings", self.stop_strings)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                stop_strings=list(stop_strings),
                tokenizer=self._tokenizer,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
                **config,
            )
        prefix_len = inputs["input_ids"].shape[-1]
        texts = self._decode(output.sequences[:, prefix_len:])
        if getattr(output, "sequences_scores", None) is not None:
            scores = output.sequences_scores.float().cpu().tolist()
        else:
            scores = [0.0] * len(texts)
        return list(zip([self._finish_step(t) for t in texts], scores))


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
