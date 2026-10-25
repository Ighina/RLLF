"""GoedelProbePolicy with a stub backend: feature order (anchors outer, layers inner) and the
logistic-regression read-out must match the offline fit (`probe_eval2.feats`)."""
import numpy as np
import pytest

from rllf.policy.probe_policy import GoedelProbePolicy
from rllf.state import PolicyState


class _Anchors:
    def __init__(self, arrays):
        self.anchors = arrays


class StubBackend:
    """Deterministic residual stream: anchor a, layer l, dim k -> hash-free constant pattern."""
    def __init__(self, layers, d=6):
        self.capture_layers, self.d = list(layers), d
        self.calls = []

    def prompt(self, claim):
        return "PROMPT " + claim

    def step_span(self, prompt, step):
        return [1, 2], len(prompt.split())

    def capture(self, prompts, weights):
        self.calls.append((prompts, weights))
        out = {}
        for a, wlist in weights.items():
            arr = np.zeros((len(prompts), len(self.capture_layers), self.d), dtype=np.float16)
            for b in range(len(prompts)):
                for li, l in enumerate(self.capture_layers):
                    arr[b, li, :] = (1 if a == "prompt_last" else 10) * l + np.arange(self.d) / 10 + b
            out[a] = arr
        return _Anchors(out)


def test_multilayer_probe_readout_matches_offline_feature_order(tmp_path):
    layers, anchors, d = [2, 5], ["prompt_last", "step_mean"], 6
    rng = np.random.default_rng(0)
    nfeat = len(layers) * len(anchors) * d
    mu, sd, coef, b = rng.normal(size=nfeat).astype(np.float32), (rng.random(nfeat) + 0.5).astype(np.float32), rng.normal(size=nfeat).astype(np.float32), 0.3
    np.savez(tmp_path / "p.npz", layers=np.array(layers), anchors=np.array(anchors), rendering="bare", mu=mu, sd=sd, coef_std=coef, intercept=b)
    backend = StubBackend(layers, d)
    pol = GoedelProbePolicy(str(tmp_path / "p.npz"), backend=backend)
    assert pol.layers == layers and pol.anchors == anchors and pol.rendering == "bare"
    states = [PolicyState("q", ["h"], "step one"), PolicyState("q", [], "step two")]
    got = pol.raw_logits(states)
    fo = backend.capture(["x", "y"], {a: [None, None] for a in anchors})
    feats = np.concatenate([np.asarray(fo.anchors[a][:, li, :], dtype=np.float32) for a in anchors for li in range(len(layers))], axis=1)
    want = ((feats - mu) / sd) @ coef + b
    assert got == pytest.approx(want, rel=1e-4)
    assert pol.calls == 2
    # bare rendering: the claim is the step itself
    assert backend.calls[0][0][0] == "PROMPT step one"
