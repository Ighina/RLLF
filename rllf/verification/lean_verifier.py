"""Lean-backed step verification.

    natural-language step -> auto-formalization -> Lean -> r_step in {0, 1}

Reward is binary, as specified, but the richer diagnostics (parse error,
elaboration error, false proposition, timeout, unsupported claim, which tactic
closed the goal) are retained on the result for use as auxiliary signal.

Practical concerns from the methodology are handled here: results are cached on
a hash of `question + previous steps + current step`, every compile runs in an
isolated temporary directory under a timeout, and independent steps can be
verified concurrently.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (
    FORMALIZATION_FAILED,
    LEAN_CHECK_FAILED,
    VERIFIED,
    StepVerification,
    StepVerifier,
)
from .formalizer import FORMALIZER_VERSION, Formalizer, RuleBasedFormalizer

# Diagnostic categories (auxiliary signal; the reward itself stays binary).
CAT_OK = "ok"
CAT_PARSE_ERROR = "parse_error"
CAT_ELABORATION_ERROR = "elaboration_error"
CAT_UNSOLVED_GOAL = "unsolved_goal"
CAT_FALSE_PROPOSITION = "false_proposition"
CAT_TIMEOUT = "timeout"
CAT_TOOLCHAIN_MISSING = "toolchain_missing"
CAT_HEADER_ERROR = "header_error"

_MESSAGE_RE = re.compile(
    r"^(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning|information):\s*",
    re.MULTILINE,
)


@dataclass
class LeanConfig:
    """How to invoke Lean, and with which proof search ladder."""

    lean_cmd: Tuple[str, ...] = ("lean",)
    #: Lake project supplying the environment (e.g. a project depending on
    #: mathlib).  When set, Lean runs as `lake env lean` from that directory,
    #: which reuses the project's already-built oleans instead of re-elaborating.
    project_dir: Optional[str] = None
    header: str = ""
    #: Tactic ladder, tried in order; the first that closes the goal wins.
    tactics: Tuple[str, ...] = ("decide", "rfl", "omega", "norm_num", "simp", "native_decide")
    #: Tactics that only exist once mathlib is on the path.  They are dropped
    #: automatically otherwise, so a bare toolchain does not spend a line of
    #: the ladder on an "unknown tactic" error.
    mathlib_tactics: Tuple[str, ...] = ("norm_num", "linarith", "nlinarith", "ring_nf", "positivity", "polyrith")
    timeout: float = 30.0
    #: `native_decide` evaluates the proposition with the compiler and seals
    #: the result behind a generated axiom, so a step it closes is *not*
    #: kernel-checked (`kernel_checked` records this per step).  Disabling it
    #: gives a kernel-only reward -- but note that without mathlib no remaining
    #: tactic can decide ground `Rat` arithmetic, so every division or decimal
    #: claim would then score 0 for want of a checker rather than for being
    #: wrong.  Such steps are flagged `uncheckable_regime` in the diagnostics.
    allow_native_decide: bool = True
    max_workers: int = 4
    extra_args: Tuple[str, ...] = ()

    def command(self, filename: str) -> List[str]:
        if self.project_dir:
            return ["lake", "env", *self.lean_cmd, *self.extra_args, filename]
        return [*self.lean_cmd, *self.extra_args, filename]

    @property
    def has_mathlib(self) -> bool:
        return self.project_dir is not None or "Mathlib" in self.header

    def effective_tactics(self, hints: Sequence[str] = ()) -> List[str]:
        """Hinted tactics first, then the rest of the ladder, deduplicated."""
        ordered: List[str] = []
        for tactic in list(hints) + list(self.tactics):
            if tactic in ordered:
                continue
            if tactic == "native_decide" and not self.allow_native_decide:
                continue
            if tactic in self.mathlib_tactics and not self.has_mathlib:
                continue
            ordered.append(tactic)
        return ordered


@dataclass
class LeanMessage:
    line: int
    column: int
    severity: str
    text: str


@dataclass
class LeanResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    messages: List[LeanMessage] = field(default_factory=list)


class LeanRunner:
    """Runs Lean source in an isolated temporary directory."""

    def __init__(self, config: Optional[LeanConfig] = None):
        self.config = config or LeanConfig()

    @property
    def available(self) -> bool:
        binary = "lake" if self.config.project_dir else self.config.lean_cmd[0]
        return shutil.which(binary) is not None

    def run(self, source: str) -> LeanResult:
        cwd = self.config.project_dir or None
        workdir = tempfile.mkdtemp(prefix="rllf-lean-", dir=cwd)
        path = Path(workdir) / "Snippet.lean"
        path.write_text(source, encoding="utf-8")
        # Deterministic compiler settings: no colour, fixed locale, no editing
        # of the caller's environment.
        env = dict(os.environ, NO_COLOR="1", LEAN_NO_COLOR="1", LC_ALL="C")
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                self.config.command(str(path)),
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=cwd,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return LeanResult(False, -1, "", "timeout", time.perf_counter() - start, True)
        except FileNotFoundError as exc:
            return LeanResult(False, -2, "", str(exc), time.perf_counter() - start)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        output = (proc.stdout or "") + (proc.stderr or "")
        return LeanResult(
            ok=proc.returncode == 0,
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            duration=time.perf_counter() - start,
            messages=parse_lean_messages(output),
        )


def parse_lean_messages(output: str) -> List[LeanMessage]:
    """Parse `file:line:col: severity: text` diagnostics out of Lean output."""
    messages: List[LeanMessage] = []
    matches = list(_MESSAGE_RE.finditer(output))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        messages.append(
            LeanMessage(
                line=int(match.group("line")),
                column=int(match.group("col")),
                severity=match.group("sev"),
                text=output[match.end() : end].strip(),
            )
        )
    return messages


def classify_error(text: str) -> str:
    lowered = text.lower()
    # `decide` says "proved that ..."; `native_decide` says "evaluated that ...".
    if "that the proposition" in lowered and "is false" in lowered:
        return CAT_FALSE_PROPOSITION
    if "unsolved goals" in lowered:
        return CAT_UNSOLVED_GOAL
    # `omega could not prove the goal`, `linarith failed`, ... -- the statement
    # elaborated fine, the tactic just did not close it.
    if "could not prove" in lowered or "tactic failed" in lowered or "failed to prove" in lowered:
        return CAT_UNSOLVED_GOAL
    if "unexpected token" in lowered or "expected" in lowered and "token" in lowered:
        return CAT_PARSE_ERROR
    if "unknown identifier" in lowered or "unknown constant" in lowered:
        return CAT_ELABORATION_ERROR
    if "failed to synthesize" in lowered or "type mismatch" in lowered:
        return CAT_ELABORATION_ERROR
    if "timeout" in lowered or "maximum recursion" in lowered or "deterministic" in lowered:
        return CAT_TIMEOUT
    return CAT_ELABORATION_ERROR


class VerificationCache:
    """Content-addressed cache of verification results.

    Keyed on `question + previous steps + current step`, as recommended, so a
    step re-generated during resampling is never re-compiled.
    """

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else None
        self._memory: Dict[str, Dict[str, Any]] = {}
        if self.path and self.path.exists():
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        self._memory[record["key"]] = record["value"]

    @staticmethod
    def make_key(question: str, previous_steps: Sequence[str], step: str, salt: str = "") -> str:
        payload = json.dumps(
            [question, list(previous_steps), step, salt], ensure_ascii=False, sort_keys=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._memory.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        self._memory[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._memory)


class LeanStepVerifier(StepVerifier):
    """Auto-formalize a step, then ask Lean whether it holds."""

    def __init__(
        self,
        formalizer: Optional[Formalizer] = None,
        config: Optional[LeanConfig] = None,
        cache: Optional[VerificationCache] = None,
        runner: Optional[LeanRunner] = None,
    ):
        self.formalizer = formalizer or RuleBasedFormalizer()
        self.config = config or LeanConfig()
        self.runner = runner or LeanRunner(self.config)
        self.cache = cache if cache is not None else VerificationCache()
        self._pool: Optional[ThreadPoolExecutor] = None

    # -- public API -------------------------------------------------------
    def verify(
        self, question: str, previous_steps: Sequence[str], current_step: str
    ) -> StepVerification:
        key = VerificationCache.make_key(
            question, previous_steps, current_step, salt=self._cache_salt()
        )
        cached = self.cache.get(key)
        if cached is not None:
            return StepVerification(
                status=cached["status"],
                reward=cached["reward"],
                formalization=cached.get("formalization"),
                error=cached.get("error"),
                diagnostics={**cached.get("diagnostics", {}), "cached": True},
            )
        result = self._verify_uncached(question, previous_steps, current_step)
        self.cache.put(key, result.to_dict())
        return result

    def verify_many(
        self, question: str, previous_steps: Sequence[str], steps: Sequence[str]
    ) -> List[StepVerification]:
        """Verify sibling candidate steps concurrently (one Lean process each)."""
        if len(steps) <= 1 or self.config.max_workers <= 1:
            return [self.verify(question, previous_steps, s) for s in steps]
        pool = self._ensure_pool()
        futures = [pool.submit(self.verify, question, previous_steps, s) for s in steps]
        return [f.result() for f in futures]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # -- internals --------------------------------------------------------
    def _ensure_pool(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return self._pool

    def _uncheckable_regime(self, formalization: Any, tactics: Sequence[str]) -> bool:
        """True when no tactic in the ladder can decide this class of claim.

        A 0 produced here says "the checker cannot express this", not "the step
        is wrong" -- a distinction worth keeping out of the training labels.
        """
        if formalization.diagnostics.get("numeric_type") != "Rat":
            return False
        if formalization.diagnostics.get("n_variables"):
            return False
        return "native_decide" not in tactics and not self.config.has_mathlib

    def _cache_salt(self) -> str:
        """Invalidate cache entries when the checking regime changes."""
        return json.dumps(
            [
                self.config.header,
                list(self.config.tactics),
                self.config.allow_native_decide,
                self.config.project_dir,
                type(self.formalizer).__name__,
                getattr(self.formalizer, "version", FORMALIZER_VERSION),
            ],
            sort_keys=True,
        )

    def _verify_uncached(
        self, question: str, previous_steps: Sequence[str], current_step: str
    ) -> StepVerification:
        formalization = self.formalizer.formalize(question, previous_steps, current_step)
        if not formalization.ok or not formalization.statement:
            return StepVerification(
                status=FORMALIZATION_FAILED,
                reward=0.0,
                formalization=None,
                error=formalization.error,
                diagnostics={**formalization.diagnostics, "category": formalization.error},
            )

        if not self.runner.available:
            return StepVerification(
                status=LEAN_CHECK_FAILED,
                reward=0.0,
                formalization=formalization.statement,
                error="lean toolchain not found",
                diagnostics={"category": CAT_TOOLCHAIN_MISSING},
            )

        tactics = self.config.effective_tactics(formalization.tactic_hints)
        uncheckable = self._uncheckable_regime(formalization, tactics)
        source, tactic_lines = build_source(
            statement=formalization.statement,
            tactics=tactics,
            header=self.config.header,
            preamble=formalization.preamble,
        )
        result = self.runner.run(source)
        diagnostics: Dict[str, Any] = {
            **formalization.diagnostics,
            "uncheckable_regime": uncheckable,
            "tactics_tried": tactics,
            "duration_s": round(result.duration, 3),
            "parser_ok": True,
            "n_errors": sum(1 for m in result.messages if m.severity == "error"),
        }

        if result.timed_out:
            diagnostics["category"] = CAT_TIMEOUT
            return StepVerification(
                status=LEAN_CHECK_FAILED,
                reward=0.0,
                formalization=formalization.statement,
                error="lean timed out",
                diagnostics=diagnostics,
            )

        errors_by_line: Dict[int, List[LeanMessage]] = {}
        for message in result.messages:
            if message.severity == "error":
                errors_by_line.setdefault(message.line, []).append(message)

        header_lines = {line for line in errors_by_line if line < min(tactic_lines.values())}
        if header_lines:
            diagnostics["category"] = CAT_HEADER_ERROR
            first = errors_by_line[min(header_lines)][0]
            return StepVerification(
                status=LEAN_CHECK_FAILED,
                reward=0.0,
                formalization=formalization.statement,
                error=first.text[:400],
                diagnostics=diagnostics,
            )

        for tactic in tactics:
            if not errors_by_line.get(tactic_lines[tactic]):
                diagnostics.update({"category": CAT_OK, "tactic": tactic, "kernel_checked": tactic != "native_decide"})
                return StepVerification(
                    status=VERIFIED,
                    reward=1.0,
                    formalization=f"example : {formalization.statement} := by {tactic}",
                    diagnostics=diagnostics,
                )

        classified = [
            (classify_error(m.text), m)
            for messages in errors_by_line.values()
            for m in messages
        ]
        categories = [category for category, _ in classified]
        # A refuted proposition is the strongest evidence available: the step is
        # not merely unproven, it is false.  Report the message that says so,
        # rather than whichever tactic happened to fail first.
        category = CAT_UNSOLVED_GOAL
        reported = None
        for candidate in (CAT_FALSE_PROPOSITION, *categories):
            match = next((m for c, m in classified if c == candidate), None)
            if match is not None:
                category, reported = candidate, match
                break
        diagnostics.update({"category": category, "error_categories": categories})
        return StepVerification(
            status=LEAN_CHECK_FAILED,
            reward=0.0,
            formalization=formalization.statement,
            error=(reported.text[:400] if reported else "no proof found"),
            diagnostics=diagnostics,
        )


def build_source(
    statement: str,
    tactics: Sequence[str],
    header: str = "",
    preamble: str = "",
) -> Tuple[str, Dict[str, int]]:
    """Emit one `example` per tactic and record each one's line number.

    All attempts share a single compilation: Lean reports errors per line, so
    the ladder costs one process instead of one per tactic.
    """
    lines: List[str] = []
    if header.strip():
        lines.extend(header.strip().splitlines())
    if preamble.strip():
        lines.extend(preamble.strip().splitlines())
    lines.append("set_option maxHeartbeats 400000")
    tactic_lines: Dict[str, int] = {}
    for tactic in tactics:
        lines.append(f"example : {statement} := by {tactic}")
        tactic_lines[tactic] = len(lines)  # 1-indexed, matches Lean diagnostics
    return "\n".join(lines) + "\n", tactic_lines


class MockStepVerifier(StepVerifier):
    """Deterministic stand-in for Lean, for tests and offline pipeline runs.

    Uses the same formalizer, then evaluates the extracted arithmetic claim in
    Python.  It exercises every code path the real verifier does -- including
    `formalization_failed` -- without requiring a toolchain.
    """

    def __init__(self, formalizer: Optional[Formalizer] = None):
        self.formalizer = formalizer or RuleBasedFormalizer()

    def verify(
        self, question: str, previous_steps: Sequence[str], current_step: str
    ) -> StepVerification:
        from .formalizer import extract_relations

        formalization = self.formalizer.formalize(question, previous_steps, current_step)
        if not formalization.ok:
            return StepVerification(
                status=FORMALIZATION_FAILED,
                reward=0.0,
                error=formalization.error,
                diagnostics={**formalization.diagnostics, "category": formalization.error},
            )
        relations = extract_relations(current_step)
        conclusion = relations[-1] if relations else None
        if conclusion is None:
            return StepVerification(
                status=LEAN_CHECK_FAILED,
                reward=0.0,
                formalization=formalization.statement,
                error="no conclusion to evaluate",
                diagnostics={"category": CAT_UNSOLVED_GOAL, "mock": True},
            )
        if conclusion.is_numeric:
            holds, error = _eval_relation(conclusion.lhs, conclusion.op, conclusion.rhs)
        else:
            # Mirror what `omega` does in Lean: discharge the conclusion using
            # the defining equations from this step and the history.
            hypotheses = relations[:-1] + [
                rel for text in previous_steps for rel in extract_relations(text)
            ]
            holds, error = _eval_implication(hypotheses, conclusion)
        if error:
            return StepVerification(
                status=LEAN_CHECK_FAILED,
                reward=0.0,
                formalization=formalization.statement,
                error=error,
                diagnostics={"category": CAT_PARSE_ERROR, "mock": True},
            )
        return StepVerification(
            status=VERIFIED if holds else LEAN_CHECK_FAILED,
            reward=1.0 if holds else 0.0,
            formalization=formalization.statement,
            error=None if holds else "claim evaluates to false",
            diagnostics={
                "category": CAT_OK if holds else CAT_FALSE_PROPOSITION,
                "mock": True,
                **formalization.diagnostics,
            },
        )


def _eval_relation(lhs: str, op: str, rhs: str) -> Tuple[bool, Optional[str]]:
    """Evaluate a ground arithmetic relation with exact rational arithmetic."""
    from fractions import Fraction

    try:
        from sympy import Rational, sympify

        left = sympify(lhs, rational=True)
        right = sympify(rhs, rational=True)
        if left.free_symbols or right.free_symbols:
            return False, "claim contains free variables"
        left, right = Rational(left), Rational(right)
    except Exception:  # sympy missing or parse failure -> Fraction fallback
        try:
            left = _eval_fraction(lhs)
            right = _eval_fraction(rhs)
        except Exception as exc:
            return False, f"cannot evaluate claim: {exc}"
        left, right = Fraction(left), Fraction(right)
    comparisons = {
        "=": left == right,
        "!=": left != right,
        "<": left < right,
        "<=": left <= right,
        ">": left > right,
        ">=": left >= right,
    }
    return bool(comparisons[op]), None


def _eval_implication(hypotheses, conclusion) -> Tuple[bool, Optional[str]]:
    """Solve the hypothesis equations, then evaluate the conclusion."""
    try:
        from sympy import Eq, solve, sympify
    except ImportError:  # pragma: no cover - sympy is a hard dependency of the mock
        return False, "sympy required to evaluate conditional claims"
    try:
        equations = [
            Eq(sympify(h.lhs, rational=True), sympify(h.rhs, rational=True))
            for h in hypotheses
            if h.op == "="
        ]
        if not equations:
            return False, "no defining equation for the conclusion variables"
        solution = solve(equations, dict=True)
        if not solution:
            return False, "hypotheses have no solution"
        lhs = sympify(conclusion.lhs, rational=True).subs(solution[0])
        rhs = sympify(conclusion.rhs, rational=True).subs(solution[0])
        if lhs.free_symbols or rhs.free_symbols:
            return False, "conclusion is not determined by the hypotheses"
    except Exception as exc:
        return False, f"cannot evaluate claim: {exc}"
    return _eval_relation(str(lhs), conclusion.op, str(rhs))


def _eval_fraction(expr: str):
    from fractions import Fraction

    if not re.fullmatch(r"[\d\.\s\+\-\*/\^\(\)]+", expr):
        raise ValueError(f"non-numeric expression: {expr!r}")
    python_expr = re.sub(r"(\d+\.\d+|\d+)", r"Fraction('\1')", expr.replace("^", "**"))
    return eval(python_expr, {"__builtins__": {}}, {"Fraction": Fraction})  # noqa: S307
