"""Guided decoding on a real (frozen) Hugging Face model.

    python examples/huggingface_base_model.py --model Qwen/Qwen2.5-0.5B-Instruct

Nothing about the base model is fine-tuned: it only proposes candidate steps.
The process reward model reranks them, and Lean supplies the reward signal.
"""

from __future__ import annotations

import argparse

from rllf.generation.base_model import GenerationConfig, HFBaseModel
from rllf.generation.guided_decoder import MODE_REJECTION, VERIFY_ONLINE, DecodeConfig, GuidedDecoder
from rllf.pipeline import GeneratorSpec, Problem, TrajectoryCollector, dataset_summary
from rllf.policy.scoring import load_policy
from rllf.verification.lean_verifier import LeanConfig, LeanStepVerifier, VerificationCache
from rllf.verification.outcome_checker import default_outcome_checker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--policy", help="directory produced by `train.py train`")
    parser.add_argument("--lean-project", help="lake project directory, for mathlib")
    parser.add_argument("--collect", action="store_true", help="collect training trajectories instead of decoding")
    args = parser.parse_args()

    base_model = HFBaseModel(args.model, generation=GenerationConfig(temperature=0.8, max_new_tokens=96))
    verifier = LeanStepVerifier(
        config=LeanConfig(project_dir=args.lean_project, header="import Mathlib" if args.lean_project else ""),
        cache=VerificationCache("runs/lean_cache.jsonl"),
    )
    checker = default_outcome_checker()

    problems = [
        Problem(
            question="A shop has 7 boxes of pens with 6 pens in each box. It gives away 5 pens. How many pens are left?",
            reference="37",
        )
    ]

    if args.collect:
        # Stage 1: build the dataset D = {(q, tau, r_step, r_outcome)}.
        collector = TrajectoryCollector(
            [GeneratorSpec(name=args.model, model=base_model, num_samples=4, temperature=0.9)],
            verifier,
            checker,
        )
        trajectories = collector.collect(problems, output_path="runs/hf_trajectories.jsonl", progress=True)
        print(dataset_summary(trajectories))
        return

    # Stages 2-3: policy reranking with verification-aware rejection sampling.
    policy = load_policy(args.policy) if args.policy else None
    decoder = GuidedDecoder(
        base_model,
        policy=policy,
        verifier=verifier,
        outcome_checker=checker,
        config=DecodeConfig(mode=MODE_REJECTION, num_candidates=4, alpha=1.0, verification=VERIFY_ONLINE),
    )
    result = decoder.generate(problems[0].question, problems[0].reference)
    for index, step in enumerate(result.trajectory.steps):
        print(f"[{index}] verified={step.verified} {step.text}")
        if step.formalization:
            print(f"      lean: {step.formalization}")
    print(f"final answer: {result.final_answer} (outcome reward {result.outcome})")
    print(result.stats.to_dict())


if __name__ == "__main__":
    main()
