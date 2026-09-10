"""Natural language -> Lean auto-formalization.

Step 1 of the step-level reward: extract the mathematical claim from a
natural-language reasoning step and translate it into Lean syntax.  Two
implementations are provided:

* :class:`RuleBasedFormalizer` -- deterministic, dependency-light, covers the
  arithmetic/linear-algebraic claims that dominate grade-school and
  competition word problems.
* :class:`LLMFormalizer` -- asks a language model to translate, for claims the
  rule-based extractor rejects.

Both return a :class:`Formalization`; the Lean verifier turns that into a file
and runs the compiler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Bumped whenever extraction or rendering changes in a way that alters the
#: Lean statement produced for some step.  It is part of the verification cache
#: key, so a behaviour change invalidates stale verdicts instead of silently
#: reusing them -- the class name alone cannot detect that.
FORMALIZER_VERSION = 3

# Statuses used in Formalization.error / diagnostics.
NO_CLAIM = "no_claim_found"
UNPARSEABLE = "unparseable_claim"
UNSUPPORTED = "unsupported_claim"


@dataclass
class Formalization:
    """A Lean proposition extracted from one natural-language step."""

    statement: Optional[str] = None
    claim: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None
    preamble: str = ""
    #: Tactics worth trying first for this statement shape, most likely first.
    tactic_hints: Tuple[str, ...] = ()
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class Formalizer:
    """Interface: natural-language step -> Lean proposition."""

    def formalize(
        self, question: str, previous_steps: Sequence[str], step: str
    ) -> Formalization:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_LATEX_REPLACEMENTS: Tuple[Tuple[str, str], ...] = (
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", " "),
    (r"\;", " "),
    (r"\\times", "*"),
    (r"\\cdot", "*"),
    (r"\\div", "/"),
    (r"\\pm", "+"),
    (r"\\leq", "<="),
    (r"\\le\b", "<="),
    (r"\\geq", ">="),
    (r"\\ge\b", ">="),
    (r"\\neq", "!="),
    (r"\\ne\b", "!="),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
)

_UNICODE_REPLACEMENTS = {
    "×": "*",
    "·": "*",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "≤": "<=",
    "≥": ">=",
    "≠": "!=",
    "＝": "=",
}

_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
#: Juxtaposition is multiplication in maths prose (`4B`, `4(3)`, `(a)(b)`) but
#: is a syntax error in Lean, so it is made explicit.
_IMPLICIT_MUL_RES = (
    # Only true juxtaposition (no space): `4 B` in prose is two words, not a
    # product, and inserting `*` there would invent claims out of English.
    re.compile(r"(?<=\d)(?=[A-Za-z])"),        # 4B     -> 4 * B
    # The left operand must be a numeral or a closing paren.  An identifier
    # followed by `(` is function application -- `f(x)`, `g(n+1)` -- and
    # rewriting that to `f * (x)` produces Lean that elaborates happily while
    # meaning something else entirely, which is the one failure mode a
    # verifier-based reward cannot detect.
    re.compile(r"(?<=[\d\)])(?=\()"),           # 4(3), (a)(b) -> explicit *
    re.compile(r"(?<=\))(?=[\dA-Za-z_])"),       # (3)x -> (3) * x
)
_SQRT_PLACEHOLDER = "\x00sqrt\x00"
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_NUM_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_CURRENCY_RE = re.compile(r"[$£€]")


def normalize_math_text(text: str) -> str:
    """Turn LaTeX-flavoured prose into plain infix arithmetic."""
    out = text
    for pattern, repl in _LATEX_REPLACEMENTS:
        out = re.sub(pattern, repl, out)
    for _ in range(3):  # nested fractions
        new = _FRAC_RE.sub(r"((\1)/(\2))", out)
        if new == out:
            break
        out = new
    out = _SQRT_RE.sub(_SQRT_PLACEHOLDER + r" (\1)", out)
    for src, dst in _UNICODE_REPLACEMENTS.items():
        out = out.replace(src, dst)
    out = out.replace("$", " ").replace("\\(", " ").replace("\\)", " ")
    out = out.replace("\\[", " ").replace("\\]", " ")
    out = _CURRENCY_RE.sub(" ", out)
    out = _NUM_COMMA_RE.sub("", out)
    out = out.replace("**", "^")
    for pattern in _IMPLICIT_MUL_RES:
        out = pattern.sub(" * ", out)
    return out.replace(_SQRT_PLACEHOLDER, "Real.sqrt")


# --------------------------------------------------------------------------
# Claim extraction
# --------------------------------------------------------------------------

_RELATION_RE = re.compile(r"(<=|>=|!=|<|>|=)")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)?|[A-Za-z_][A-Za-z_0-9]*|<=|>=|!=|[<>=]|[\+\-\*/\^\(\)]|\S"
)

#: Short English words that would otherwise be mistaken for variables.
_STOPWORDS = frozenset(
    """a an as at be by do go he if in is it me my no of on or so to up us we
    am are and the for was has had its her him not now one two six ten""".split()
)

#: Identifiers allowed inside a math span despite being long.
_ALLOWED_WORDS = {"Real.sqrt"}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?$")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*$")
_OPERATORS = set("+-*/^()")


@dataclass
class Relation:
    """One relational claim `lhs op rhs` recovered from the text."""

    lhs: str
    op: str
    rhs: str
    raw: str

    @property
    def lean_op(self) -> str:
        return {"=": "=", "!=": "\u2260", "<=": "\u2264", ">=": "\u2265", "<": "<", ">": ">"}[self.op]

    def lean(self) -> str:
        return f"{self.lhs} {self.lean_op} {self.rhs}"

    @property
    def is_numeric(self) -> bool:
        return not free_symbols(self.raw)


def free_symbols(expr: str) -> List[str]:
    """Variable names appearing in an arithmetic expression."""
    names: List[str] = []
    for word in _WORD_RE.findall(expr):
        if word in _ALLOWED_WORDS or "." in word:
            continue
        if word not in names:
            names.append(word)
    return names


def _is_variable_token(token: str) -> bool:
    """Variables are short identifiers (`x`, `y2`, `n`), never English words."""
    return (
        _IDENT_RE.match(token) is not None
        and len(token) <= 2
        and token.lower() not in _STOPWORDS
    )


def _math_spans(text: str) -> List[str]:
    """Maximal runs of arithmetic tokens, cut at prose words and punctuation."""
    spans: List[str] = []
    current: List[Tuple[int, int]] = []

    def flush() -> None:
        if current:
            start, end = current[0][0], current[-1][1]
            spans.append(text[start:end])
            current.clear()

    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        is_math = (
            _NUMBER_RE.match(token) is not None
            or token in _OPERATORS
            or _RELATION_RE.fullmatch(token) is not None
            or _is_variable_token(token)
        )
        if is_math:
            current.append(match.span())
        else:
            flush()
    flush()
    return spans


def _balanced(expr: str) -> bool:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def extract_relations(text: str) -> List[Relation]:
    """Pull every well-formed relational claim out of a step.

    Chained claims (`3 * 4 = 12 = 12`) yield one `Relation` per comparison.
    """
    relations: List[Relation] = []
    for span in _math_spans(normalize_math_text(text)):
        if not _RELATION_RE.search(span) or not re.search(r"\d", span):
            continue
        parts = _RELATION_RE.split(span)
        operands = [p.strip() for p in parts[0::2]]
        operators = parts[1::2]
        for i, op in enumerate(operators):
            lhs, rhs = operands[i], operands[i + 1]
            if _is_expression(lhs) and _is_expression(rhs):
                relations.append(Relation(lhs, op, rhs, f"{lhs} {op} {rhs}"))
    return relations


def _is_expression(expr: str) -> bool:
    """Cheap well-formedness check for an infix arithmetic expression."""
    expr = expr.strip()
    if not expr:
        return False
    if not re.fullmatch(r"[0-9A-Za-z_\.\+\-\*/\^\(\)\s]+", expr):
        return False
    if re.search(r"[\+\*/\^]\s*[\+\*/\^]", expr):
        return False
    if expr[-1] in "+-*/^":
        return False
    return _balanced(expr)


# --------------------------------------------------------------------------
# Lean rendering
# --------------------------------------------------------------------------

_DECIMAL_RE = re.compile(r"\b\d+\.\d+\b")


def choose_numeric_type(expressions: Sequence[str]) -> str:
    """Pick the smallest Lean numeric type that keeps the claim faithful.

    `Nat` truncates subtraction and division, so any expression containing
    `-` or `/` is lifted to `Int`/`Rat`.  Getting this wrong is the classic way
    to obtain a Lean-checkable statement that does not mean what the prose said.
    """
    joined = " ".join(expressions)
    if "/" in joined or _DECIMAL_RE.search(joined):
        return "Rat"
    if "-" in joined:
        return "Int"
    return "Nat"


def render_statement(
    relations: Sequence[Relation],
    conclusion: Relation,
    numeric_type: str,
) -> str:
    """Render `∀ vars, hyps → conclusion` in Lean syntax."""
    variables: List[str] = []
    for rel in list(relations) + [conclusion]:
        for name in free_symbols(rel.raw):
            if name not in variables:
                variables.append(name)

    body = conclusion.lean()
    for hypothesis in reversed(list(relations)):
        body = f"{hypothesis.lean()} → {body}"

    if variables:
        binder = " ".join(variables)
        return f"∀ ({binder} : {numeric_type}), {body}"
    # Ascribe the type on a ground claim so Lean elaborates in `numeric_type`.
    return f"({conclusion.lhs} : {numeric_type}) {conclusion.lean_op} {conclusion.rhs}"


class RuleBasedFormalizer(Formalizer):
    """Deterministic extractor for arithmetic and linear claims.

    The extracted string is rendered from the *original* text, never from a
    symbolic simplification of it: simplifying `3 + 4 = 7` before handing it to
    Lean would turn the check into a tautology and destroy the reward signal.
    """

    def __init__(self, max_hypotheses: int = 3, use_history: bool = True):
        self.max_hypotheses = max_hypotheses
        self.use_history = use_history

    def formalize(
        self, question: str, previous_steps: Sequence[str], step: str
    ) -> Formalization:
        relations = extract_relations(step)
        if not relations:
            return Formalization(
                ok=False,
                error=NO_CLAIM,
                diagnostics={"reason": "no relational claim in step"},
            )

        conclusion = relations[-1]
        hypotheses = list(relations[:-1])[-self.max_hypotheses :]

        needed = set(free_symbols(conclusion.raw))
        if needed and self.use_history:
            hypotheses = self._augment_with_history(previous_steps, hypotheses, needed)

        bound = set()
        for rel in hypotheses:
            bound.update(free_symbols(rel.raw))
        unbound = needed - bound
        if unbound and not conclusion.is_numeric:
            # A free variable with no defining hypothesis makes the universally
            # quantified reading false; report it rather than fabricate a proof.
            return Formalization(
                ok=False,
                error=UNSUPPORTED,
                claim=conclusion.raw,
                diagnostics={"unbound_variables": sorted(unbound)},
            )

        exprs = [r.raw for r in hypotheses] + [conclusion.raw]
        numeric_type = choose_numeric_type(exprs)
        statement = render_statement(hypotheses, conclusion, numeric_type)
        # A ground claim is rendered without its hypotheses (see
        # `render_statement`), so key the tactic ladder off the conclusion.
        has_variables = bool(needed)
        return Formalization(
            statement=statement,
            claim=conclusion.raw,
            ok=True,
            tactic_hints=_tactic_hints(numeric_type, has_variables),
            diagnostics={
                "numeric_type": numeric_type,
                "n_hypotheses": len(hypotheses),
                "n_variables": len(free_symbols(" ".join(exprs))),
                "formalizer": "rule_based",
            },
        )

    def _augment_with_history(
        self,
        previous_steps: Sequence[str],
        hypotheses: List[Relation],
        needed: set,
    ) -> List[Relation]:
        """Borrow defining equations for variables from earlier steps."""
        known = {name for rel in hypotheses for name in free_symbols(rel.raw)}
        missing = needed - known
        if not missing:
            return hypotheses
        extra: List[Relation] = []
        for text in reversed(list(previous_steps)):
            for rel in reversed(extract_relations(text)):
                names = set(free_symbols(rel.raw))
                if names & missing:
                    extra.append(rel)
                    missing -= names
            if not missing:
                break
        return list(reversed(extra))[-self.max_hypotheses :] + hypotheses


def _tactic_hints(numeric_type: str, has_variables: bool) -> Tuple[str, ...]:
    if has_variables:
        # `simp_arith` is deprecated (a hard error since Lean 4.2x); `omega`
        # covers the linear Nat/Int goals it used to serve.
        return ("omega", "decide", "norm_num", "simp")
    if numeric_type == "Rat":
        return ("norm_num", "native_decide", "decide", "rfl")
    return ("decide", "norm_num", "rfl", "omega", "native_decide")


class LLMFormalizer(Formalizer):
    """Fallback formalizer that asks a language model for the Lean statement.

    `generator` must expose ``generate(prompt, max_new_tokens=..., ...) -> str``
    (:class:`~rllf.generation.base_model.BaseModelAdapter` satisfies this).
    """

    PROMPT = (
        "Translate the final mathematical claim of the reasoning step into a "
        "single self-contained Lean 4 proposition.\n"
        "Reply with the proposition only, no `theorem`/`example` keyword and no proof.\n\n"
        "Question: {question}\n"
        "Previous steps:\n{history}\n"
        "Step: {step}\n"
        "Lean proposition:"
    )

    def __init__(self, generator: Any, fallback: Optional[Formalizer] = None, **gen_kwargs: Any):
        self.generator = generator
        self.fallback = fallback
        self.gen_kwargs = {"max_new_tokens": 96, "temperature": 0.0, **gen_kwargs}

    def formalize(
        self, question: str, previous_steps: Sequence[str], step: str
    ) -> Formalization:
        prompt = self.PROMPT.format(
            question=question,
            history="\n".join(f"- {s}" for s in previous_steps) or "- (none)",
            step=step,
        )
        try:
            raw = self.generator.generate(prompt, **self.gen_kwargs)
        except Exception as exc:  # pragma: no cover - depends on backend
            return self._fallback(question, previous_steps, step, str(exc))
        statement = _clean_lean_statement(raw)
        if not statement:
            return self._fallback(question, previous_steps, step, "empty generation")
        return Formalization(
            statement=statement,
            claim=step,
            ok=True,
            tactic_hints=("norm_num", "decide", "omega", "simp", "native_decide"),
            diagnostics={"formalizer": "llm"},
        )

    def _fallback(
        self, question: str, previous_steps: Sequence[str], step: str, error: str
    ) -> Formalization:
        if self.fallback is not None:
            result = self.fallback.formalize(question, previous_steps, step)
            result.diagnostics.setdefault("llm_error", error)
            return result
        return Formalization(ok=False, error=UNPARSEABLE, diagnostics={"llm_error": error})


def _clean_lean_statement(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        blocks = re.findall(r"```(?:lean4?)?\n?(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[0]
    text = text.strip().splitlines()[0].strip() if text.strip() else ""
    text = re.sub(r"^(theorem|example|lemma)\b[^:]*:", "", text).strip()
    text = text.split(":=")[0].strip()
    return text
