# What the Lean machinery actually is

**Short answer: none of the `lean4:*` skills, agents, or lean-lsp MCP tools are
involved.** The `lean4` plugin loaded into my session at startup is tooling for
*me* to write and debug proofs interactively. This repository needs something
different: an unattended, batch, timeout-bounded oracle that turns a sentence of
English into a 0 or a 1, thousands of times, inside a training loop.

So `rllf/verification/lean_verifier.py` talks to the Lean toolchain the only way
that survives that setting — **one `subprocess.run` of the `lean` binary on a
generated temp file, with the compiler's diagnostics parsed back out of stdout.**

```bash
$ grep -rn "lean-lsp\|mcp__\|autoformalize\|lakefile" rllf/ train.py tests/
(no matches)
```

The repository has no lakefile, no `lean-toolchain`, and no `.lake` directory.
Its only requirement is a `lean` on the `PATH`; here that is elan's shim at
`~/.elan/bin/lean`, Lean **4.33.1**. Everything degrades to `--mock-verifier`
(the same claims evaluated in exact rational arithmetic) when it is absent, and
`LeanRunner.available` is what decides.

Why not the skills: they are interactive and conversational. A skill invocation
costs a model turn, cannot be called from inside `TrajectoryCollector.annotate`,
cannot be run 4-wide from a thread pool, has no timeout contract, and would make
the reward signal depend on a language model's judgement — which is precisely
the dependency the methodology exists to remove. The reward has to come from the
compiler.

---

## The pipeline

```
  natural-language step
        │
   (1)  ├─ normalise LaTeX / unicode            normalize_math_text
   (2)  ├─ scan tokens into math spans          _math_spans
   (3)  ├─ split spans into relations           extract_relations
   (4)  ├─ bind free variables from history     _augment_with_history
   (5)  ├─ choose the numeric type              choose_numeric_type
   (6)  └─ render a Lean proposition            render_statement
        │
        ▼  Formalization(statement, tactic_hints, diagnostics)
        │
   (7)  ├─ assemble one file, one `example` per tactic   build_source
   (8)  ├─ run `lean` in an isolated temp dir            LeanRunner.run
   (9)  ├─ parse `file:line:col: severity:` messages     parse_lean_messages
  (10)  └─ attribute errors to lines → winning tactic    _verify_uncached
        │
        ▼  StepVerification(status, reward ∈ {0,1}, formalization, diagnostics)
        │
  (11)  └─ memoise on sha256(question + history + step + regime)
```

Stages 1–6 are the "auto-formalization" box in the methodology; 7–10 are the
"Lean verification" box. They live in `formalizer.py` and `lean_verifier.py`
respectively, and `LLMFormalizer` can replace 1–6 wholesale.

---

## Stages 1–6: natural language → a Lean proposition

### (1) Normalisation — `formalizer.py:95`

LaTeX and unicode collapse to plain infix arithmetic before anything else looks
at the text. `\frac{a}{b}` → `((a)/(b))` (three nested passes), `\times`/`·`/`×`
→ `*`, `\le`/`≤` → `<=`, `−` (U+2212) → `-`, `$`/`\(`/`\[` → space, currency
symbols dropped, `1,000` → `1000`, `**` → `^`.

### (2) Token scan — `formalizer.py:181`

The step is scanned into tokens, each classified as *math* (a number, an
operator, a relation, or a **short** identifier) or *prose*. Maximal runs of
math tokens become candidate spans; prose cuts a run.

This is the part that stops the extractor from hallucinating equations out of
English. A regex that merely looks for `=` turns

> "Then the total **cost = 12** dollars."

into a claim. Here `total` and `cost` are prose (length > 2), so the run is cut
and only `= 12` survives — which has an empty left operand and is discarded. A
stopword list handles the short words that would otherwise pass as variables
(`so`, `is`, `of`, `we`, `at`, `an`, `a`, …).

### (3) Relations — `formalizer.py:220`

Each span is split on its relation symbols, so a chain yields one `Relation`
per comparison: `"3 * 4 = 12 and 12 + 1 = 13"` → `[3 * 4 = 12, 12 + 1 = 13]`.
Both operands must pass a well-formedness check (balanced parens, no doubled
operators, no trailing operator). **The last relation in the step is the
conclusion; the earlier ones become hypotheses.**

### (4) Variable binding — `formalizer.py:361`

If the conclusion mentions a variable that nothing in the step defines, the
formalizer walks *backwards through the step history* for a defining equation.
A variable that stays unbound is reported as `unsupported_claim` — it is **not**
universally quantified, because `∀ z, z = 4` is false and would hand the policy
a fabricated negative label.

### (5) Numeric type — `formalizer.py:260`

`Nat` truncates: `5 - 8 = 0` and `10 / 4 = 2` are *true* in `Nat` and would let
an arithmetic error compile. So any expression containing `-` becomes `Int`, and
anything with `/` or a decimal becomes `Rat`. This is the concrete defence
against §14's "compiles but says something else".

### (6) Rendering — `formalizer.py:275`

```
ground claim      →  (3 * 4 : Nat) = 12
conditional claim →  ∀ (x : Nat), x + 2 = 5 → x = 3
from history      →  ∀ (y : Int), y - 3 = 5 → y = 8
```

The type ascription sits on the left operand so Lean elaborates the whole
equation in the chosen type. Ground claims are rendered **without** their
sibling hypotheses (each is independently checkable); only variable-bearing
claims become implications.

> **The expression is rendered from the normalised source text, never from a
> symbolic simplification of it.** Passing `3 * 4 = 12` through sympy would
> yield `12 = 12`, Lean would close it by `rfl`, and every step in the dataset —
> right or wrong — would earn reward 1. The single most important property of
> this stage is that it does no arithmetic.

---

## Stages 7–10: the proposition → a bit

### (7) One file, one `example` per tactic — `lean_verifier.py:430`

There is no proof search. There is a fixed ladder of tactics, and the file
tries all of them at once:

```lean
set_option maxHeartbeats 400000
example : (50 / 5 : Rat) = 10 := by native_decide
example : (50 / 5 : Rat) = 10 := by decide
example : (50 / 5 : Rat) = 10 := by rfl
example : (50 / 5 : Rat) = 10 := by omega
example : (50 / 5 : Rat) = 10 := by simp
```

`build_source` returns the source **and the line number of each tactic**
(`{'native_decide': 2, 'decide': 3, 'rfl': 4, 'omega': 5, 'simp': 6}`).

Because Lean reports errors with line positions and keeps elaborating after a
failed `example`, a *single* compilation tells us the outcome of every tactic.
The ladder therefore costs one process (~0.4 s) rather than one process per
tactic (~2 s). Hints from the formalizer are ordered first
(`LeanConfig.effective_tactics`), so the likely winner is usually line 2.

### (8) Execution — `lean_verifier.py:136`

```python
workdir = tempfile.mkdtemp(prefix="rllf-lean-", dir=project_dir or None)
(workdir/"Snippet.lean").write_text(source)
env = dict(os.environ, NO_COLOR="1", LEAN_NO_COLOR="1", LC_ALL="C")
subprocess.run(["lean", str(path)], capture_output=True, text=True,
               timeout=self.config.timeout, cwd=cwd, env=env)
# finally: shutil.rmtree(workdir)
```

* **Isolation** — a fresh temp directory per check, removed in `finally`, so a
  crashed or timed-out compile leaves nothing behind.
* **Timeout** — `subprocess.TimeoutExpired` becomes `category: timeout`, never a
  hang. `maxHeartbeats` is a second, in-process bound.
* **Determinism** — colour off and `LC_ALL=C` so the diagnostic strings that
  `classify_error` matches on do not shift with locale or TTY.
* **No mutation of the caller's environment**; `os.environ` is copied, not edited.
* When `project_dir` is set the command becomes `lake env lean …` run from that
  directory, which reuses the project's already-built `.olean` files instead of
  re-elaborating its imports.

### (9) Diagnostics — `lean_verifier.py:171`

```python
_MESSAGE_RE = r"^(?P<file>[^\s:]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<sev>error|warning|information):\s*"
```

stdout and stderr are concatenated and split on this anchor; each message keeps
everything up to the next anchor, so multi-line goal states survive intact.

### (10) Attribution — `lean_verifier.py:327`

1. Errors are bucketed by line.
2. An error on a line *before* the first tactic is a header failure (a bad
   `import`), reported as `header_error` — not as a property of the step.
3. Tactics are walked in ladder order; **the first whose line carries no error
   won**, and its name and the full `example … := by <tactic>` are stored on the
   `StepRecord` for audit.
4. If none won, every error is classified and the *strongest* verdict is
   reported, preferring `false_proposition` — being refuted is stronger evidence
   than merely being unproven, and the message reported is the one that says so
   rather than whichever tactic happened to fail first.

`classify_error` matches on `"that the proposition … is false"` — which covers
both `decide`'s "proved that…" and `native_decide`'s "evaluated that…" —
`"unsolved goals"`, `"unexpected token"`, `"unknown identifier/constant"`,
`"failed to synthesize"`/`"type mismatch"`, and deterministic-timeout wording.

### (11) Cache — `lean_verifier.py:206`

`sha256(json([question, previous_steps, step, salt]))`, where the salt is
`[header, tactics, allow_native_decide, project_dir, formalizer class]`. Keying
on the salt means changing the tactic ladder or adding mathlib invalidates the
old verdicts instead of silently reusing them. Backed by an append-only JSONL
file and read into memory at construction.

Measured on this machine (Lean 4.33.1, M-series, no mathlib):

| | 4 distinct steps |
| --- | --- |
| cold, sequential | 1.66 s (~0.4 s/step) |
| cold, 4 threads | 0.79 s |
| cached | 0.05 ms |

Concurrency is a thread pool (`verify_many`) — the work is in a subprocess, so
the GIL is irrelevant.

---

## The tactic ladder and what it costs in trust

Default ladder: `decide`, `rfl`, `omega`, `norm_num`, `simp`, `native_decide`,
filtered by `effective_tactics`:

* mathlib-only tactics (`norm_num`, `linarith`, `ring_nf`, …) are **dropped when
  mathlib is not on the path**, so a bare toolchain does not burn a ladder line
  on `unknown tactic`;
* `native_decide` is dropped when `allow_native_decide=False`.

Axiom footprint, measured with `#print axioms`:

| tactic | closes | axioms |
| --- | --- | --- |
| `decide` | ground `Nat`/`Int` arithmetic | *none* — pure kernel reduction |
| `omega` | linear `Nat`/`Int`, including implications from hypotheses | `propext`, `Quot.sound` |
| `rfl`/`simp` | definitional/simp-normal goals | standard |
| `native_decide` | ground `Rat`, large numerals | `propext`, `Classical.choice`, `Quot.sound`, **plus a generated `…native_decide.ax_…`** |

That last row is the honest caveat: `native_decide` compiles the proposition to
native code, evaluates it, and seals the result behind an axiom. **A step it
closes has been checked by the compiler, not by the kernel.** Every verified
record carries `diagnostics["kernel_checked"]` recording which happened.

### The consequence nobody mentions until it bites

Without mathlib, `native_decide` is the *only* tactic in the ladder that can
decide ground `Rat` arithmetic. Verified directly:

```
example : (50 / 5 : Rat) = 10 := by decide   -- Decidable instance did not reduce
example : (50 / 5 : Rat) = 10 := by simp     -- `simp` made no progress
example : (50 / 5 : Rat) = 10 := by rfl      -- 50 / 5 is not definitionally equal to 10
example : (50 / 5 : Rat) = 10 := by native_decide  -- ✓
```

So setting `allow_native_decide=False` on a bare toolchain would score **every
division and decimal step 0 — for want of a checker, not for being wrong** —
and quietly teach the policy that division is bad reasoning. Rather than hide
this, the verifier detects the situation and flags it: `diagnostics
["uncheckable_regime"] = True` (`lean_verifier.py:302`). A 0 carrying that flag
means "the checker cannot express this claim", and can be filtered out of the
training labels. Installing mathlib (`norm_num`) removes the dilemma.

---

## Result taxonomy

The reward stays binary, as specified. The diagnostics do not.

| status | reward | `category` | meaning |
| --- | --- | --- | --- |
| `verified` | 1 | `ok` | a tactic closed the goal (`tactic`, `kernel_checked` recorded) |
| `formalization_failed` | 0 | `no_claim_found` | no relational claim in the step |
| `formalization_failed` | 0 | `unsupported_claim` | free variable with no defining hypothesis |
| `lean_check_failed` | 0 | `false_proposition` | Lean **refuted** the claim — the step is wrong |
| `lean_check_failed` | 0 | `unsolved_goal` | true or false, but no tactic in the ladder closed it |
| `lean_check_failed` | 0 | `parse_error` / `elaboration_error` | the rendered statement was not well-formed Lean |
| `lean_check_failed` | 0 | `timeout` | wall-clock or heartbeat limit |
| `lean_check_failed` | 0 | `header_error` | the preamble failed (e.g. a missing `import`) |
| `lean_check_failed` | 0 | `toolchain_missing` | no `lean` on the `PATH` |

`false_proposition` versus `unsolved_goal` is the distinction that matters most
downstream: the first is evidence about the *step*, the second is evidence about
the *checker*. Both currently earn 0, and both are recorded so the split can be
revisited without regenerating the dataset.

---

## Worked example

Input step, with the previous step as history:

```
history = ["Each of the 7 boxes holds 6 pens, so 7 * 6 = 42."]
step    = "After giving away 5 pens, 42 - 5 = 36."
```

| stage | result |
| --- | --- |
| normalise | unchanged (no LaTeX) |
| spans | `['5', '42 - 5 = 36']` — `After`/`giving`/`away`/`pens` cut the runs; the bare `5` carries no relation and is dropped |
| relations | `[42 - 5 = 36]`; last one is the conclusion, no hypotheses |
| variables | none → ground claim |
| type | contains `-` → `Int` |
| statement | `(42 - 5 : Int) = 36` |
| ladder | `decide`, `rfl`, `omega`, `native_decide`, `simp` — hints first, `norm_num` filtered out (no mathlib) |
| Lean | line 2: `Tactic `decide` proved that the proposition 42 - 5 = 36 is false` |
| result | `lean_check_failed`, reward **0**, `category: false_proposition` |

Change the step to `= 37` and line 2 comes back clean: `verified`, reward **1**,
`tactic: decide`, `kernel_checked: True`.

---

## Enabling mathlib

```bash
python train.py generate --lean-project /path/to/lake-project --lean-header "import Mathlib"
```

`has_mathlib` then reports true, so `norm_num` re-enters the ladder ahead of
`native_decide`; commands run as `lake env lean` from the project root, reusing
its built `.olean` files. Expect the per-step cost to rise (importing Mathlib is
seconds, not milliseconds) — which is what the cache and the `hybrid`
verification schedule exist to absorb.

---

## What this deliberately is not

* **Not a proof search.** A fixed ladder, no backtracking, no tactic
  generation. Any step needing a real proof lands in `unsolved_goal`. The
  interface is a `StepVerifier`, so a search-based prover drops straight in.
* **Not a semantic-faithfulness check.** Lean confirms the *rendered* statement,
  never that it means what the English said. That is why the exact statement is
  stored on every `StepRecord`, why the type choice is conservative, and why
  §14's "a verifier reward is not a correctness reward" is repeated in the code.
* **Not the only formalizer.** `LLMFormalizer` (`formalizer.py:394`) sends the
  step to a language model, strips fences and any `theorem …:=`, and falls back
  to the rule-based extractor on failure. Stages 7–11 are unchanged — the rest of
  the system only ever sees a `Formalization`.
* **Not required.** `MockStepVerifier` reproduces the same three statuses by
  evaluating the extracted claim with `sympy`/`Fraction` (and solving hypothesis
  equations, mirroring `omega`), so tests and CI need no toolchain. Real Lean is
  exercised by the tests guarded with `@requires_lean`.

## Where to look

| concern | file |
| --- | --- |
| NL → Lean statement | `rllf/verification/formalizer.py` |
| Lean invocation, ladder, cache, diagnostics | `rllf/verification/lean_verifier.py` |
| Statuses and the adapter interfaces | `rllf/verification/base.py` |
| Final-answer checking (separate signal) | `rllf/verification/outcome_checker.py` |
| Tests, incl. real-Lean ones | `tests/test_verification.py` |
