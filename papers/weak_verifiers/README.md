# Paper B — Weak verifiers for reasoning steps (in progress)

Working folder. Sources of truth: `forked-tree-classifier-retraining.md` §11.3, §13 and `results_consolidated.md` §E.

## Thesis (provisional)
A code model writing Python checks is a cheap second verifier whose *refutations* carry the correctness signal Lean
lacks; verifiers should be routed (Python always, Lean on router-selected steps), and a generator can be taught to tag
each step with the verifier that applies, which helps accuracy on hard problems.

## Done
- Weak verifier on 20,964 steps, two checkers, sycophancy control (§13.1)
- Python-verifiability probes and classifiers (§13.2, §13.4); two-stage router (§13.3)
- Routed-verifier beam search with Lean labels and an independent re-check (§13.5)
- Verifiable-CoT SFT round 1 and round 2 (§13.6, §13.7)
- Trajectory rerank by real Python verdicts on 260 fresh hard problems (§11.3)

## Running / queued
- Second seed + second evaluation split for the tag effect; base model on both splits (`launch_vcot2_seed.sh`)
- OpenJev zero-shot three-way router on all labelled steps (`openjev_router.py`)

## Notes
- Context in verification is already the default: Lean gets previous steps as hypotheses (no problem text); Python gets
  the problem, the previous steps and the claim.
- Baseline for every SFT claim is the un-fine-tuned Qwen2.5-7B-Instruct (0.845 easy / 0.527 hard greedy).
