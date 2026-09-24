"""Theorem 5: the compressed additive estimator is unbiased with O(k^-1/2) error."""
import numpy as np
import pytest

from ecnca.sketch import ThetaSketch


def _trial(n_roots, k, seed, dim=4):
    rng = np.random.default_rng(seed)
    sk = ThetaSketch(dim, k, hash_seed=seed)   # hash seed = the sketch's randomness
    total = np.zeros(dim)
    for i in range(n_roots):
        p = rng.gamma(2.0, 1.0, size=dim)      # positive, heavy-ish tail
        total += p
        sk.insert(f"root::{seed}::{i}", p, float(p.sum()))
    return sk.estimate_payload(), total, sk.estimate_count()


@pytest.mark.parametrize("k", [16, 64])
def test_payload_estimator_unbiased(k):
    n, trials = 400, 400
    rel = []
    for s in range(trials):
        est, true, _ = _trial(n, k, s)
        rel.append(est.sum() / true.sum() - 1.0)
    rel = np.array(rel)
    se = rel.std(ddof=1) / np.sqrt(trials)
    assert abs(rel.mean()) < 4 * se + 1e-3, f"bias {rel.mean():.4f} vs se {se:.4f}"


def test_error_decays_as_sqrt_k():
    n, trials = 512, 300
    stds = {}
    for k in (8, 32, 128):
        rel = [_trial(n, k, s)[0].sum() / _trial(n, k, s)[1].sum() - 1.0 for s in range(trials)]
        stds[k] = float(np.std(rel, ddof=1))
    # a 16x increase in k should shrink the spread by roughly 4x
    ratio = stds[8] / stds[128]
    assert 2.0 < ratio < 8.0, stds


def test_count_estimator_unbiased():
    n, k, trials = 1000, 32, 400
    ests = [_trial(n, k, s)[2] for s in range(trials)]
    m = float(np.mean(ests))
    se = float(np.std(ests, ddof=1)) / np.sqrt(trials)
    assert abs(m - n) < 4 * se, (m, se)
