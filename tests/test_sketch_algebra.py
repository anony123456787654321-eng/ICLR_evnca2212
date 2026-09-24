"""Theorem 1: exact provenance merging is commutative, associative, idempotent.

These are the property tests that Gate 2 requires.  They are the reason
message order, cycles and re-delivery cannot change an EC cell's fixed point.
"""
import numpy as np
import pytest

from ecnca.sketch import ExactLedger, ThetaSketch

KINDS = [("exact", lambda d: ExactLedger(d)),
         ("theta4", lambda d: ThetaSketch(d, 4)),
         ("theta32", lambda d: ThetaSketch(d, 32)),
         ("theta32_cons", lambda d: ThetaSketch(d, 32, rescale=False)),
         ("theta_split", lambda d: ThetaSketch(d, 24, hash_capacity=160))]


def _fill(sk, roots, rng):
    for r in roots:
        sk.insert(r, rng.normal(size=sk.payload_dim), 1.0)
    return sk


def _roots(rng, n, tag):
    return [f"{tag}::{i}" for i in range(n)]


@pytest.mark.parametrize("name,ctor", KINDS)
@pytest.mark.parametrize("n", [3, 40, 200])
def test_idempotent(name, ctor, n):
    rng = np.random.default_rng(0)
    a = _fill(ctor(6), _roots(rng, n, "a"), rng)
    assert a.merge(a) == a
    assert a.merge(a).merge(a) == a


@pytest.mark.parametrize("name,ctor", KINDS)
@pytest.mark.parametrize("n", [3, 40, 200])
def test_commutative(name, ctor, n):
    rng = np.random.default_rng(1)
    a = _fill(ctor(6), _roots(rng, n, "a"), rng)
    b = _fill(ctor(6), _roots(rng, n, "b"), rng)
    assert a.merge(b) == b.merge(a)


@pytest.mark.parametrize("name,ctor", KINDS)
@pytest.mark.parametrize("n", [3, 40, 200])
def test_associative(name, ctor, n):
    rng = np.random.default_rng(2)
    a = _fill(ctor(6), _roots(rng, n, "a"), rng)
    b = _fill(ctor(6), _roots(rng, n, "b"), rng)
    c = _fill(ctor(6), _roots(rng, n, "c"), rng)
    assert a.merge(b).merge(c) == a.merge(b.merge(c))


@pytest.mark.parametrize("n", [1, 8, 32])
def test_theta_exact_below_capacity(n):
    """Theorem: at most k roots => theta stays 1 => estimates are exact."""
    rng = np.random.default_rng(3)
    sk = ThetaSketch(6, 32)
    total = np.zeros(6)
    for r in _roots(rng, n, "r"):
        p = rng.normal(size=6)
        total += p
        sk.insert(r, p, 1.0)
    assert sk.is_exact
    np.testing.assert_allclose(sk.estimate_payload(), total, rtol=1e-12)
    assert sk.estimate_count() == n


@pytest.mark.parametrize("k", [4, 16, 64])
def test_repeated_root_adds_nothing(k):
    """Theorem 4: repeating one root any number of times adds zero evidence."""
    rng = np.random.default_rng(4)
    sk = ThetaSketch(6, k)
    for r in _roots(rng, 3 * k, "r"):
        sk.insert(r, rng.normal(size=6), 1.0)
    before = (sk.estimate_payload().copy(), sk.estimate_mass(), sk.estimate_count())
    for _ in range(100):
        for r in _roots(rng, 3 * k, "r"):
            sk.insert(r, np.full(6, 1e6), 1e6)   # same roots, absurd claimed mass
    # a same-refinement re-delivery may never raise the credited information
    assert sk.estimate_mass() <= before[1] + 1e-9
    np.testing.assert_allclose(sk.estimate_payload(), before[0], rtol=1e-12)
    assert sk.estimate_mass() == pytest.approx(before[1])
    assert sk.estimate_count() == pytest.approx(before[2])


def test_order_invariance_under_shuffle():
    rng = np.random.default_rng(5)
    roots = _roots(rng, 500, "r")
    payloads = {r: rng.normal(size=6) for r in roots}
    sigs = []
    for trial in range(8):
        order = list(roots)
        np.random.default_rng(trial).shuffle(order)
        sk = ThetaSketch(6, 32)
        for r in order:
            sk.insert(r, payloads[r], 1.0)
        sigs.append(sk.signature())
    assert len(set(sigs)) == 1


def test_conservative_readout_is_a_subset_posterior():
    """rescale=False must report the EXACT sum of the roots it retained.

    That is what makes the conservative belief calibrated without a variance
    correction: it is a real posterior conditioned on a lineage-distinct
    subsample, not an extrapolation from one.
    """
    rng = np.random.default_rng(11)
    payloads = {f"r::{i}": rng.normal(size=6) for i in range(400)}
    sk = ThetaSketch(6, 32, rescale=False)
    for r, p in payloads.items():
        sk.insert(r, p, 1.0)
    retained = np.sum([payloads[r] for r in sk.entries], axis=0)
    np.testing.assert_allclose(sk.estimate_payload(), retained, rtol=1e-12)
    assert not np.any(sk.estimate_payload_cov())   # no estimation variance to report


def test_conservative_retention_is_deterministic_across_cells():
    """Two cells that saw the same roots in different orders retain the SAME set.

    This is the property an LRU or reservoir subsample lacks, and the reason the
    conservative readout survives merging.
    """
    rng = np.random.default_rng(12)
    payloads = {f"r::{i}": rng.normal(size=6) for i in range(500)}
    sets = []
    for trial in range(6):
        order = list(payloads)
        np.random.default_rng(trial).shuffle(order)
        sk = ThetaSketch(6, 32, rescale=False)
        for r in order:
            sk.insert(r, payloads[r], 1.0)
        sets.append(frozenset(sk.entries))
    assert len(set(sets)) == 1


@pytest.mark.parametrize("name,ctor", KINDS)
def test_refinement_replaces_belief_without_adding_evidence(name, ctor):
    """Behaviour 1 and 2 together, at the level of the ledger.

    A better computation over the SAME observation must be allowed to replace
    the belief -- otherwise B's abstraction of A's evidence is thrown away
    merely because it descends from A -- while the number of roots credited,
    and hence the confidence, stays exactly where it was.
    """
    rng = np.random.default_rng(21)
    sk = ctor(6)
    roots = _roots(rng, 3, "r")
    for r in roots:
        sk.insert(r, np.ones(6), 1.0, 0.25)          # crude extraction
    count_before = sk.estimate_count()
    for r in roots:
        sk.insert(r, 4.0 * np.ones(6), 4.0, 1.00)    # fully refined, same roots
    assert sk.estimate_count() == pytest.approx(count_before)   # no new evidence
    np.testing.assert_allclose(sk.estimate_payload(), 3 * 4.0 * np.ones(6), rtol=1e-9)


@pytest.mark.parametrize("name,ctor", KINDS)
def test_refinement_join_is_order_invariant(name, ctor):
    """Refinements may arrive in any order, including backwards."""
    rng = np.random.default_rng(22)
    roots = _roots(rng, 40, "r")
    levels = [0.2, 0.5, 0.9, 1.0]
    sigs = []
    for trial in range(6):
        order = [(r, f) for r in roots for f in levels]
        np.random.default_rng(trial).shuffle(order)
        sk = ctor(6)
        for r, f in order:
            sk.insert(r, f * np.ones(6), f, f)
        sigs.append(sk.signature())
    assert len(set(sigs)) == 1
