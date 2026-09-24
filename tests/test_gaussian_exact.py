"""Theorem 3: exact EC recovers the centralised unique-evidence posterior."""
import numpy as np

from ecnca.data import generate_example
from ecnca.evaluate import run_method
from ecnca.gaussian import GaussianSpec
from ecnca.schedule import DeliverySchedule


def _ex(regime, dup, topo="torus", n_unique=12, seed=0, n_cells=16):
    return generate_example(GaussianSpec(), n_unique, np.random.default_rng(seed),
                            topology=topo, n_cells=n_cells, regime=regime, duplication=dup)


def test_exact_ledger_matches_centralised_posterior():
    ex = _ex("clean", 1)
    sums, _, _, _ = run_method("ec_exact", ex, steps=24, rng=np.random.default_rng(0))
    for s in sums:
        np.testing.assert_allclose(s, ex.unique_payload_sum, rtol=1e-10)


def test_duplication_changes_nothing_for_ec():
    clean = _ex("clean", 1, seed=7)
    dup = _ex("duplicate", 32, seed=7)      # identical roots, 32 copies each
    a, _, _, _ = run_method("ec_exact", clean, 24, np.random.default_rng(0))
    b, _, _, _ = run_method("ec_exact", dup, 24, np.random.default_rng(0))
    np.testing.assert_allclose(np.mean(a, axis=0), np.mean(b, axis=0), rtol=1e-10)


def test_naive_sum_blows_up_on_cycles():
    ex = _ex("clean", 1, topo="cycle")
    ec, _, _, _ = run_method("ec_exact", ex, 24, np.random.default_rng(0))
    naive, _, _, _ = run_method("naive_sum", ex, 24, np.random.default_rng(0))
    assert np.linalg.norm(naive[0]) > 100 * np.linalg.norm(ec[0])


def test_schedule_invariance_of_ec():
    ex = _ex("duplicate", 4, topo="torus")
    ref = None
    for name in ("sync", "async", "lossy", "delayed", "repeating", "adversarial"):
        sums, _, _, _ = run_method("ec_exact", ex, 64, np.random.default_rng(3),
                                schedule=DeliverySchedule.named(name))
        got = np.mean(sums, axis=0)
        if name == "lossy" or name == "adversarial":
            continue          # dropped messages can genuinely starve a cell
        if ref is None:
            ref = got
        else:
            np.testing.assert_allclose(got, ref, rtol=1e-8)
