"""MATH-style answer equivalence: LaTeX / unicode / plain-text forms of the
same value compared symbolically.

The composite `default_outcome_checker` compared `-√3/3` (what the
step-native model writes) with `-\\frac{\\sqrt{3}}{3}` (the MATH reference) by
exact string match and graded it wrong -- every trajectory of the first
forked tree was mathematically right and scored 0/4. `normalize_math_text`
(`formalizer.py`) is aimed at Lean, not sympy, so this module does its own
LaTeX -> sympy translation and falls back to a canonical-string comparison
for things sympy has no business parsing (tuples, intervals, matrices,
multiple-choice letters).
"""

from __future__ import annotations

import re
from typing import List, Optional

_STRIP_COMMANDS = (
    r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad",
    r"^\circ", r"^{\circ}", r"\circ", "°", r"\%", "%", r"\$", "$", r"\mbox", r"\textbf",
    r"\(", r"\)", r"\[", r"\]",
)
_TEXT_RE = re.compile(r"\\(?:text|mathrm|textrm|operatorname)\s*\{([^{}]*)\}")
_UNITS_RE = re.compile(
    r"\\text\{\s*(?:degrees?|units?|cm|m|km|inches|feet|ft|dollars?|cents?|square|sq|meters?|"
    r"minutes?|hours?|seconds?|days?|years?)\s*\}"
)
_UNICODE = {
    "√": r"\sqrt", "π": r"\pi", "×": "*", "·": "*", "−": "-", "–": "-", "÷": "/", "∞": r"\infty",
    "²": "^2", "³": "^3", "½": "(1/2)", "¼": "(1/4)", "¾": "(3/4)", "θ": "theta", "α": "alpha",
    "β": "beta", "ϕ": "phi", "φ": "phi", "λ": "lambda", "μ": "mu", "σ": "sigma",
}
_FRAC_CMD_RE = re.compile(r"\\[dt]?frac")
_SQRT_BRACE_RE = re.compile(r"\\sqrt\s*\{")
_SQRT_BARE_RE = re.compile(r"\\sqrt\s*([0-9A-Za-z.]+)")
_SQRT_N_RE = re.compile(r"\\sqrt\s*\[([^\]]*)\]\s*\{")
_LATEX_FUNCS = ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp", "arcsin", "arccos", "arctan")
_NUM_COMMA_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _balanced_arg(text: str, start: int) -> Optional[tuple]:
    """`text[start] == '{'` -> (inner, index after the matching '}')."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    for pos in range(start, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : pos], pos + 1
    return None


def _expand_frac(text: str) -> str:
    while True:
        match = _FRAC_CMD_RE.search(text)
        if not match:
            return text
        pos = match.end()
        while pos < len(text) and text[pos] == " ":
            pos += 1
        first = _balanced_arg(text, pos)
        if first is None:
            # `\frac12` style: two single-character arguments
            bare = re.match(r"\s*(\S)\s*(\S)", text[match.end():])
            if not bare:
                return text
            num, den, end = bare.group(1), bare.group(2), match.end() + bare.end()
        else:
            num, after = first
            while after < len(text) and text[after] == " ":
                after += 1
            second = _balanced_arg(text, after)
            if second is None:
                return text
            den, end = second
        text = f"{text[:match.start()]}(({num})/({den})){text[end:]}"


def _expand_sqrt(text: str) -> str:
    while True:
        match = _SQRT_N_RE.search(text)
        if not match:
            break
        arg = _balanced_arg(text, match.end() - 1)
        if arg is None:
            break
        inner, end = arg
        text = f"{text[:match.start()]}(({inner})**(1/({match.group(1)}))){text[end:]}"
    while True:
        match = _SQRT_BRACE_RE.search(text)
        if not match:
            break
        arg = _balanced_arg(text, match.end() - 1)
        if arg is None:
            break
        inner, end = arg
        text = f"{text[:match.start()]}sqrt({inner}){text[end:]}"
    return _SQRT_BARE_RE.sub(r"sqrt(\1)", text)


def latex_to_sympy_text(answer: str) -> str:
    """Best-effort translation of a MATH-style answer into sympy syntax."""
    text = answer.strip()
    text = _UNITS_RE.sub("", text)
    text = _TEXT_RE.sub(r"\1", text)
    for src, dst in _UNICODE.items():
        text = text.replace(src, dst)
    for cmd in _STRIP_COMMANDS:
        text = text.replace(cmd, "")
    text = _NUM_COMMA_RE.sub("", text)
    text = _expand_frac(text)
    text = _expand_sqrt(text)
    text = text.replace(r"\pi", "pi").replace(r"\infty", "oo").replace(r"\cdot", "*").replace(r"\times", "*")
    text = text.replace(r"\div", "/").replace(r"\pm", "+")  # `\pm` answers are compared as strings anyway
    for fn in _LATEX_FUNCS:
        text = text.replace(f"\\{fn}", fn)
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("^", "**")
    return text.strip().rstrip(".").strip()


_VALUE_SPLITS = (
    re.compile(r"=(?!.*=)"),                     # last `=`: `cot(-60) = -sqrt(3)/3`
    re.compile(r"\bis\b(?!.*\bis\b)"),           # last ` is `: `the period is 2pi/5`
    re.compile(r":(?!.*:)"),                     # last `:`: `Final answer: 7`
)


def value_candidates(answer: str) -> List[str]:
    """Plausible "the value" substrings of a prose answer, most specific
    first: the whole translated text, then what follows the last `=`, the
    last ` is `, the last `:` -- in that order, deduplicated."""
    text = latex_to_sympy_text(answer)
    out = [text]
    for pattern in _VALUE_SPLITS:
        match = pattern.search(text)
        if match:
            tail = text[match.end():].strip().rstrip(".").strip()
            if tail and tail not in out:
                out.append(tail)
    return out


def _canonical_string(text: str) -> str:
    text = re.sub(r"\s+", "", text.lower())
    text = text.replace("**", "^").rstrip(".")
    return text


def _split_top_level(text: str, sep: str = ",") -> List[str]:
    parts, depth, current = [], 0, []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [p.strip() for p in parts]


def _sympy_equal(a: str, b: str) -> Optional[bool]:
    """True/False when both sides parse; None when either does not."""
    try:
        from sympy import N, simplify
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        ea = parse_expr(a, transformations=transformations, evaluate=True)
        eb = parse_expr(b, transformations=transformations, evaluate=True)
    except Exception:
        return None
    try:
        diff = simplify(ea - eb)
        if diff == 0:
            return True
        value = N(diff)
        if value.is_number and abs(complex(value)) < 1e-9:
            return True
        # simplify can miss equalities it could evaluate numerically; a
        # free-symbol difference that is not identically 0 is unequal.
        return False
    except Exception:
        return None


def math_equivalent(answer: str, reference: str) -> Optional[bool]:
    """Are two MATH-style answers the same value?

    Returns True/False, or None when neither a symbolic nor a canonical-string
    comparison could be made (empty input). Structured answers (tuples,
    intervals, matrices) are compared element-wise after translation.
    """
    if not answer.strip() or not reference.strip():
        return None
    b_text = latex_to_sympy_text(reference)
    decided: Optional[bool] = None
    for a_text in value_candidates(answer):
        if _canonical_string(a_text) == _canonical_string(b_text):
            return True
        # element-wise for `2,4` / `(1, 2)` / `[0, 1)` style answers
        a_parts, b_parts = _split_top_level(a_text.strip("()[]")), _split_top_level(b_text.strip("()[]"))
        if len(a_parts) == len(b_parts) > 1 and "matrix" not in a_text and "matrix" not in b_text:
            results = [_sympy_equal(x, y) for x, y in zip(a_parts, b_parts)]
            if all(r is True for r in results):
                return True
            if any(r is not None for r in results):
                decided = False
            continue
        result = _sympy_equal(a_text, b_text)
        if result is True:
            return True
        if result is False:
            decided = False
    return bool(decided)
