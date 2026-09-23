"""Real (not policy-predicted) formalizability: Goedel-Formalizer-V2-8B writes
a Lean statement for a reasoning step, DeepSeek-Prover-V2-7B attempts to prove
it, and a real `lake exe repl` + mathlib compile decides PROVED/not -- the
same methodology FaithformBench used to build its training labels, run fresh
on our own MATH-500 decoding output instead of FaithformBench's fixed set.

Requires a built Lean workspace (mathlib compiled, REPL built) -- see
`docs`/session notes: `lake exe cache get` then `lake build REPL`, done once
into a `--lean-workspace` directory (kept off the home quota; e.g. under
`/scratch`).

VeriForm (github.com/Ighina/VeriForm, branch `pho`) supplies the vLLM-based
DeepSeek-Prover-V2 wrapper (`autoformalization_v2.prover.DeepSeekProver`) and
the real Lean checker (`verify_lean4_file`) this module reuses; only the
Goedel formalizer (VeriForm has none) and the RLLF-facing glue are new here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERIFORM_SRC = "/home/iacopog/VeriForm/src"


def _ensure_veriform_on_path() -> None:
    import sys

    if VERIFORM_SRC not in sys.path:
        sys.path.insert(0, VERIFORM_SRC)


def patch_lean_workspace(lean_workspace: str) -> None:
    """Point VeriForm's `verify_lean4_file` at a real, built Lean workspace.

    `lean_workspace` is a *bound default argument*, fixed at module-import
    time to a path relative to VeriForm's own (unbuilt) `deepseek/lean/`
    directory. Rebinding `__defaults__` after import changes that default for
    every caller, including the `Lean4ServerProcess` workers `mp.Process`
    forks off it (fork inherits the parent's already-patched function object).
    Must run before any `Lean4ServerScheduler` is constructed.
    """
    _ensure_veriform_on_path()
    import veriform.autoformalization_v2.deepseek.prover.lean.verifier as v2_verifier
    import veriform.autoformalization.deepseek.prover.lean.verifier as v1_verifier

    for module in (v1_verifier, v2_verifier):
        defaults = list(module.verify_lean4_file.__defaults__)
        defaults[1] = lean_workspace  # (lake_path, lean_workspace, ...)
        module.verify_lean4_file.__defaults__ = tuple(defaults)


LEAN_TEMPLATE = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

-- {statement}
{lean_code}
""".strip()

_LEAN4_CODE_RE = re.compile(r"```lean4\n(.*?)\n```", re.DOTALL)
# `:=` alone is not a safe body/proof boundary: a `let x : T := v` binding
# inside the statement (routine for the multi-variable steps Goedel
# formalizes) has its own `:=` long before the real one. Anchor on `:= by`
# instead -- the actual proof-opening idiom every candidate we've seen uses,
# and never what a `let` binding is followed by.
_THEOREM_RE = re.compile(r"theorem\s+(\w+)\s*:\s*(.*?)\s*:=\s*by\b", re.DOTALL)


@dataclass
class RealFormalizationResult:
    """One step's real (Lean-checked) formalizability verdict."""

    theorem_name: str
    step: str
    lean_statement: Optional[str] = None
    lean_proof: Optional[str] = None
    proved: bool = False
    stage: str = "formalize"  # formalize | prove | proved
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class GoedelFormalizer:
    """`Goedel-LM/Goedel-Formalizer-V2-8B` -- natural language -> Lean statement.

    Prompt and extraction match the model card exactly (chat template, a
    `theorem_name`, "think before you provide the lean statement", pull the
    last ```lean4 fenced block). No proof is expected in the output; the
    formalized statement is handed to `DeepSeekProver` next.
    """

    MODEL_DIR = "Goedel-LM/Goedel-Formalizer-V2-8B"

    def __init__(self, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.15):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_DIR)
        self.model = LLM(
            self.MODEL_DIR,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=8192,  # statements only, no proof search -- short output
        )
        self.sampling_params = SamplingParams(temperature=0.9, top_p=0.95, top_k=20, max_tokens=4096)

    def _prompt(self, theorem_name: str, informal_statement: str) -> str:
        content = (
            "Please autoformalize the following natural language problem statement in Lean 4. "
            f"Use the following theorem name: {theorem_name}\n"
            f"The natural language statement is: \n{informal_statement}"
            "Think before you provide the lean statement."
        )
        chat = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _extract(text: str) -> Optional[str]:
        matches = _LEAN4_CODE_RE.findall(text)
        return matches[-1].strip() if matches else None

    def formalize_batch(self, items: Sequence[Tuple[str, str]]) -> List[Optional[str]]:
        """`items`: `(theorem_name, informal_statement)` pairs -> Lean statement or None."""
        prompts = [self._prompt(name, statement) for name, statement in items]
        outputs = self.model.generate(prompts, self.sampling_params)
        return [self._extract(out.outputs[0].text) for out in outputs]


def render_step_claim(question: str, previous_steps: Sequence[str], step: str) -> str:
    """Phrase one reasoning step as a claim to formalize, with just enough
    context (problem + prior steps) for the formalizer to resolve variables
    the step references but doesn't (re)define."""
    parts = [f"Problem: {question}"]
    if previous_steps:
        parts.append("Previous steps (context, assume correct):")
        parts.extend(f"  {i}. {s}" for i, s in enumerate(previous_steps))
    parts.append(f"Step to verify: {step}")
    return "\n".join(parts)


import functools


@functools.lru_cache()
def _gpt2_byte_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _denoise_byte_level_bpe(text: str) -> str:
    """Reverse GPT-2/byte-level-BPE's byte<->visible-unicode remapping.

    This checkpoint's vLLM decode returns `.text` still in "byte-level BPE
    display form" rather than properly detokenized UTF-8: ASCII space and
    newline show up literally as `Ġ`/`Ċ`, and every multi-byte UTF-8 math
    symbol (`∃`, `∀`, `∧`, `ℝ`, `ℕ`, `→`, ...) comes out as a run of
    individually remapped bytes (e.g. `∃` -> `âĪĥ`). This is the exact
    map HF's `ByteLevel` tokenizer uses to make raw bytes printable
    (`bytes_to_unicode` in `transformers`/`tokenizers`); undo it byte-by-byte,
    then decode the recovered bytes as UTF-8.
    """
    unicode_to_byte = {v: k for k, v in _gpt2_byte_to_unicode().items()}
    raw = bytearray()
    for ch in text:
        byte = unicode_to_byte.get(ch)
        if byte is not None:
            raw.append(byte)
        else:
            raw.extend(ch.encode("utf-8"))
    return raw.decode("utf-8", errors="replace")


def _splice_proof(statement: str, tactic_body: str) -> Optional[str]:
    """Graft a proof onto *our* untouched statement instead of trusting the
    completing model to have faithfully restated it.

    DeepSeek-Prover-V2 is free to (and does) reformat the theorem it's asked
    to complete -- dropping inferable type ascriptions, and in some cases
    changing `∧` to `→` or otherwise drifting from the original claim. Text
    -- even whitespace/byte-decode-normalized -- comparison against that
    restatement is both fragile and, worse, the wrong thing to trust: what
    must be verified is *our* claim, not whatever the model rephrased it as.
    So: find where our own statement's signature ends (`:= by`), and splice
    the model's tactic block in right after it -- the signature Lean
    actually type-checks is always ours, verbatim.
    """
    match = _THEOREM_RE.search(statement)
    if not match:
        return None
    return statement[: match.end()] + tactic_body


def _single_gpu_deepseek_prover(batch_size: int = 4) -> Any:
    """`DeepSeekProver` with `tensor_parallel_size=1` instead of the class's
    hardcoded 2 -- this box's GPUs (140GB each) don't need a 7B model split
    across two of them, and a shared job usually only holds one card free.
    Everything else (prompt construction, `prove_single`, the Lean scheduler)
    is inherited unchanged from VeriForm's `DeepSeekProver`.
    """
    from transformers import AutoTokenizer
    from veriform.autoformalization_v2.deepseek.prover.lean.verifier import Lean4ServerScheduler
    from veriform.autoformalization_v2.prover import DeepSeekProver
    from vllm import LLM, SamplingParams

    prover = DeepSeekProver.__new__(DeepSeekProver)
    prover.system_prompt = "You are an expert in mathematics and Lean 4."
    prover.tokenizer = AutoTokenizer.from_pretrained(DeepSeekProver.MODEL_DIR)
    # This box is shared and actual free VRAM on any given card fluctuates
    # with other users' jobs (observed failures at both 0.45 and 0.55 whole-card
    # fractions when the "free" card wasn't as free as `gpuq status` suggested
    # at submit time) -- size for a *small*, reliable footprint instead of a
    # fraction of the whole card. `max_model_len` matches `sampling_params`
    # below (plus prompt room), not the checkpoint's full 65536 -- KV cache is
    # sized off this number, and reserving cache for sequences several times
    # longer than any we actually generate is what starved it in the first place.
    prover.model = LLM(
        DeepSeekProver.MODEL_DIR, tensor_parallel_size=1,
        gpu_memory_utilization=0.2, max_model_len=10000,
    )
    prover.lean_pattern = re.compile(r"```lean4(.*?)```", re.DOTALL | re.IGNORECASE)
    prover.batch_size = batch_size
    prover.theorem_pattern = _THEOREM_RE
    prover.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=8000, n=batch_size)
    prover.scheduler = Lean4ServerScheduler(max_concurrent_requests=batch_size)

    # vLLM's decode for this checkpoint leaks byte-level-BPE display form
    # instead of detokenizing (see `_denoise_byte_level_bpe`) -- every
    # generated block is otherwise syntactically valid Lean, just wearing
    # GPT-2 byte-BPE punctuation. Normalize before extraction ever sees it.
    _base_parse = DeepSeekProver.parse_lean_code

    def _parse_lean_code_denoised(self, response_text: str) -> str:
        return _base_parse(self, _denoise_byte_level_bpe(response_text))

    prover.parse_lean_code = _parse_lean_code_denoised.__get__(prover, DeepSeekProver)

    # Replaces `prove_single`/`get_valid_proof`: don't ask "did the model
    # faithfully restate our theorem" (fragile, and the wrong question --
    # see `_splice_proof`) -- graft the model's tactic block onto our own
    # untouched statement and verify that.
    from veriform.autoformalization_v2.prover import LEAN_TEMPLATE as _V2_LEAN_TEMPLATE
    from veriform.autoformalization_v2.prover import LEAN_WRAPPER_TEMPLATE as _V2_PROVER_PROMPT

    def _prove_statement(self, informal: str, statement: str) -> Optional[str]:
        user_prompt = _V2_PROVER_PROMPT.format(
            lean_code=_V2_LEAN_TEMPLATE.format(lean_code=statement, statement=informal)
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True
        )
        response = self.model.generate(prompt, sampling_params=self.sampling_params)[0]

        candidates: List[str] = []
        for i, out in enumerate(response.outputs):
            try:
                lean_code = self.parse_lean_code(out.text)
            except ValueError as exc:
                print(f"    [prover] candidate {i}: parse_lean_code failed: {exc}", flush=True)
                continue
            match = _THEOREM_RE.search(lean_code)
            if not match:
                print(f"    [prover] candidate {i}: no `:= by` found in its own restatement", flush=True)
                continue
            spliced = _splice_proof(statement, lean_code[match.end() :])
            if spliced is None:
                continue
            candidates.append(spliced)
        print(f"    [prover] {len(candidates)}/{len(response.outputs)} candidates spliced cleanly", flush=True)
        if not candidates:
            return None

        # `statement` (Goedel's own output) is already a complete,
        # self-contained Lean file -- its own `import`/`open` preamble, not a
        # bare declaration. Wrapping it in `_V2_LEAN_TEMPLATE` again would
        # duplicate that preamble *after* the outer wrapper's `open`, which
        # Lean rejects outright ("invalid 'import' command, it must be used
        # in the beginning of the file") -- on every candidate, always.
        result_ids = self.scheduler.submit_all_request(
            [dict(code=c, ast=False, tactics=False) for c in candidates]
        )
        results = self.scheduler.get_all_request_outputs(result_ids)
        for i, (candidate, result) in enumerate(zip(candidates, results)):
            print(f"    [prover] candidate {i}: complete={result.get('complete')} "
                  f"errors={result.get('errors')} sorries={len(result.get('sorries', []))}", flush=True)
            if result["complete"]:
                return candidate
        return None

    prover.prove_statement = _prove_statement.__get__(prover, DeepSeekProver)
    return prover


class RealFormalizabilityScorer:
    """Ties `GoedelFormalizer` + VeriForm's `DeepSeekProver` + a real Lean
    check into one `score_steps` call over a batch of reasoning steps."""

    def __init__(
        self,
        lean_workspace: str,
        goedel_tensor_parallel_size: int = 1,
        goedel_gpu_memory_utilization: float = 0.35,
        prover_batch_size: int = 4,
    ):
        patch_lean_workspace(lean_workspace)
        _ensure_veriform_on_path()

        self.lean_workspace = lean_workspace
        self.formalizer = GoedelFormalizer(
            tensor_parallel_size=goedel_tensor_parallel_size,
            gpu_memory_utilization=goedel_gpu_memory_utilization,
        )
        self.prover = _single_gpu_deepseek_prover(batch_size=prover_batch_size)

    def close(self) -> None:
        if hasattr(self.prover, "scheduler"):
            self.prover.scheduler.close()

    def score_steps(
        self, items: Sequence[Tuple[str, Sequence[str], str]]
    ) -> List[RealFormalizationResult]:
        """`items`: `(question, previous_steps, step)` triples, one per step
        to score -- across arms/problems in one call so the two models batch."""
        names = [f"step_{i}" for i in range(len(items))]
        claims = [render_step_claim(q, hist, s) for q, hist, s in items]
        statements = self.formalizer.formalize_batch(list(zip(names, claims)))

        results: List[RealFormalizationResult] = []
        for name, (question, _hist, step), claim, statement in zip(names, items, claims, statements):
            if not statement or not _THEOREM_RE.search(statement + " :="):
                # Goedel didn't return a parseable `theorem name : body` --
                # can't hand this to the prover at all.
                results.append(
                    RealFormalizationResult(
                        theorem_name=name, step=step, lean_statement=statement,
                        proved=False, stage="formalize",
                        error="no parseable theorem statement",
                        diagnostics={"claim": claim},
                    )
                )
                continue
            try:
                proof = self.prover.prove_statement(claim, statement)
            except Exception as exc:  # pragma: no cover - depends on model output shape
                results.append(
                    RealFormalizationResult(
                        theorem_name=name, step=step, lean_statement=statement,
                        proved=False, stage="prove", error=str(exc),
                    )
                )
                continue
            proved = proof is not None
            results.append(
                RealFormalizationResult(
                    theorem_name=name, step=step, lean_statement=statement,
                    lean_proof=proof if proved else None,
                    proved=proved, stage="proved" if proved else "prove",
                )
            )
        return results
