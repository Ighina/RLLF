"""A step judge read from the *autoformalizer's own residual stream*.

`GoedelProbePolicy` wraps Goedel-Formalizer-V2-8B (the formalizer of the real-Lean
pipeline) through LeanSteer's hooked backend: for a `(question, history, step)` state it
builds the claim the pipeline would formalize (`render_step_claim`, or the bare step),
runs one prefill, reads the residual stream at one layer and one anchor (last prompt
token, or the mean over the step's tokens) and applies a logistic-regression probe
`((x - mu) / sd) @ coef + b`. The result stands in for an external classifier's step
logit: `predict_step_probability` (Platt-scaled) and `predict_step_logit_for_beam`
(`beam_calibrator`) have the same meaning as on `ProcessRewardPolicy`, so
`lean_constrained_beam_search` can use either judge unchanged.

Probe files are the `.npz` written by LeanSteer's `fit_layer` (FaithformBench-trained) or
by `probe_eval.py` (fit on this project's own Lean-labelled steps): `layer`, `mu`, `sd`,
`coef_std`, `intercept`, optional `anchor` / `rendering`.
"""
from __future__ import annotations

import sys
from typing import Any, List, Optional, Sequence

import numpy as np
import torch

from rllf.policy.scoring import TemperatureScaler
from rllf.state import PolicyState
from rllf.verification.real_lean_pipeline import render_step_claim

LEANSTEER_ROOT = "/home/iacopog/LeanSteer"


class GoedelProbePolicy:
    def __init__(self, probe_path: str, anchor: Optional[str] = None, rendering: Optional[str] = None,
                 device: str = "cuda", dtype: str = "bfloat16", backend: Any = None,
                 step_calibrator: Optional[TemperatureScaler] = None, beam_calibrator: Optional[TemperatureScaler] = None):
        if LEANSTEER_ROOT not in sys.path:
            sys.path.insert(0, LEANSTEER_ROOT)
        from leansteer.models.backend import HFBackend
        from leansteer.models.registry import get_spec
        from leansteer.run.collect import anchor_weights

        z = np.load(probe_path, allow_pickle=True)
        # component: "resid" (residual stream, default) or "heads" (per-head attention outputs, the
        # input of each block's o_proj). Features are the concatenation, anchors outer, of either
        # the listed layers (resid; as `probe_eval2.feats`) or the head_dim slices of the listed
        # (layer, head) pairs (heads; as `probe_eval_heads.head_feats`).
        self.component = str(z["component"]) if "component" in z.files else "resid"
        self.heads = [(int(l), int(h)) for l, h in np.atleast_2d(z["heads"])] if "heads" in z.files else []
        self.head_dim = int(z["head_dim"]) if "head_dim" in z.files else None
        if self.component == "heads":
            self.layers = sorted({l for l, _ in self.heads})
        elif "layers" in z.files:
            self.layers = [int(x) for x in np.atleast_1d(z["layers"])]
        else:
            self.layers = [int(z["layer"])]
        self.anchors = ([str(x) for x in np.atleast_1d(z["anchors"])] if "anchors" in z.files
                        else [anchor or (str(z["anchor"]) if "anchor" in z.files else "step_mean")])
        self.layer, self.anchor = self.layers[0], self.anchors[0]
        self.mu = np.asarray(z["mu"], dtype=np.float32)
        self.sd = np.asarray(z["sd"], dtype=np.float32)
        self.coef = np.asarray(z["coef_std"], dtype=np.float32)
        self.intercept = float(z["intercept"])
        self.rendering = rendering or (str(z["rendering"]) if "rendering" in z.files else "facts")
        self._anchor_weights = anchor_weights
        self.backend = backend or HFBackend(get_spec("goedel"), dtype=dtype, device=device, capture_layers=self.layers)
        self.backend.capture_layers = list(self.layers)
        self.step_calibrator = step_calibrator or TemperatureScaler()
        self.beam_calibrator = beam_calibrator or TemperatureScaler()
        self.calls = 0

    # -- scoring ----------------------------------------------------------
    def claim(self, state: PolicyState) -> str:
        if self.rendering == "bare":
            return state.current_step
        return render_step_claim(state.question, list(state.history), state.current_step, style="facts")

    @torch.no_grad()
    def raw_logits(self, states: Sequence[PolicyState]) -> np.ndarray:
        prompts, weights = [], {a: [] for a in self.anchors}
        for s in states:
            p = self.backend.prompt(self.claim(s))
            idx, n = self.backend.step_span(p, s.current_step)
            prompts.append(p)
            for a in self.anchors:
                weights[a].append(self._anchor_weights(a, n, idx))
        if self.component == "heads":
            fo = self.backend.capture(prompts, weights, component="heads")        # anchors[a]: [B, n_layers, n_heads*head_dim]
            li_of = {l: i for i, l in enumerate(self.layers)}; dh = self.head_dim
            x = np.concatenate([np.asarray(fo.anchors[a][:, li_of[l], h * dh:(h + 1) * dh], dtype=np.float32)
                                for a in self.anchors for l, h in self.heads], axis=1)
        else:
            fo = self.backend.capture(prompts, weights)                           # anchors[a]: [B, n_layers, d]
            x = np.concatenate([np.asarray(fo.anchors[a][:, li, :], dtype=np.float32)
                                for a in self.anchors for li in range(len(self.layers))], axis=1)
        self.calls += len(states)
        return ((x - self.mu) / self.sd) @ self.coef + self.intercept

    def predict_step_probability(self, state: PolicyState) -> float:
        logit = torch.tensor(self.raw_logits([state]), dtype=torch.float32)
        return float(self.step_calibrator.apply(logit)[0])

    def predict_step_logit_for_beam(self, state: PolicyState) -> float:
        logit = torch.tensor(self.raw_logits([state]), dtype=torch.float32)
        return float(self.beam_calibrator.apply_logits(logit)[0])

    # -- calibration from stored scores -------------------------------------
    @staticmethod
    def fit_calibrators(raw_logits: np.ndarray, labels: np.ndarray):
        """Platt scaling for probabilities (`step_calibrator`) and the beam scale (`beam_calibrator`),
        both from raw probe logits of held-out labelled steps; the two are the same fit here --
        kept separate for parity with `ProcessRewardPolicy`."""
        l = torch.tensor(np.asarray(raw_logits, dtype=np.float32)); y = torch.tensor(np.asarray(labels, dtype=np.float32))
        return TemperatureScaler().fit(l, y), TemperatureScaler().fit(l, y)
