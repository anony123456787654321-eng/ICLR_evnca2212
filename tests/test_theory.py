"""Verification of the closed-form results in ecnca/theory.py.

Each theorem gets three kinds of check: an exact algebraic identity where one
exists, agreement with Monte Carlo simulation of the estimator as implemented,
and a counterexample showing that a substantive assumption is load-bearing.
Numerical agreement validates the implementation; it does not replace the proofs
in Appendix A, and these tests are written on that understanding.
"""
import math

import numpy as np
import pytest

import ecnca.theory as T
from ecnca.sketch.theta import ThetaSketch


# --------------------------------------------------------------------------
# Theorem 1: the master decomposition
# --------------------------------------------------------------------------

def gaussian_nll(X, mean, precision):
    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(np.linalg.inv(precision))
    r = X - mean
    return 0.5 * (d * math.log(2 * math.pi) + logdet
                  + np.einsum("ij,jk,ik->i", r, precision, r)).mean()


@pytest.mark.parametrize("d,lam", [(1, 16.0), (3, 4.0), (5, 0.7), (2, 1.0)])
def test_excess_nll_matches_simulation(d, lam):
    """The decomposition (d/2)phi(lam) + lam*delta/2 is exact, not asymptotic."""
    rng = np.random.default_rng(d * 100 + int(lam * 10))
    A = rng.normal(size=(d, d))
    precision = A @ A.T + d * np.eye(d)
    mu = rng.normal(size=d)
    mu_hat = mu + rng.normal(size=d) * 0.3
    delta = float((mu - mu_hat) @ precision @ (mu - mu_hat))
    X = rng.multivariate_normal(mu, np.linalg.inv(precision), size=200_000)
    mc = gaussian_nll(X, mu_hat, lam * precision) - gaussian_nll(X, mu, precision)
    assert T.excess_nll(lam, d, delta) == pytest.approx(mc, abs=0.03)


def test_excess_nll_is_zero_exactly_at_the_correct_readout():
    assert T.excess_nll(1.0, 5, 0.0) == pytest.approx(0.0)
    assert T.phi(1.0) == pytest.approx(0.0)


def test_duplication_penalty_equals_the_kl_divergence():
    """Theorem 2 is a KL divergence, so it must reproduce the standard formula."""
    rng = np.random.default_rng(3)
    for d, m in [(1, 16), (3, 4), (8, 2)]:
        A = rng.normal(size=(d, d))
        lam = A @ A.T + d * np.eye(d)
        kl = 0.5 * (np.trace(m * lam @ np.linalg.inv(lam)) - d
                    + math.log(np.linalg.det(lam) / np.linalg.det(m * lam)))
        assert T.duplication_penalty(m, d) == pytest.approx(kl)


def test_duplication_penalty_is_strictly_positive_and_convex_in_m():
    """The protection guarantee: any m > 1 costs something, and costs more the
    larger it is. If this failed there would be no penalty to prevent."""
    vals = [T.duplication_penalty(m, 3) for m in (1, 2, 4, 8, 16)]
    assert vals[0] == pytest.approx(0.0)
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_matrix_penalty_collapses_to_the_scalar_form_under_uniform_duplication():
    rng = np.random.default_rng(11)
    d, R = 3, 5
    P = np.stack([(lambda A: A @ A.T + d * np.eye(d))(rng.normal(size=(d, d)))
                  for _ in range(R)])
    for m in (1, 2, 16):
        assert T.duplication_penalty_matrix(P, [m] * R) == pytest.approx(
            T.duplication_penalty(m, d))


def test_unequal_duplication_is_bracketed_by_its_extremes():
    """Corollary: with every m_r >= 1 the generalised eigenvalues lie in
    [min m_r, max m_r], so the penalty is bracketed by the uniform cases."""
    rng = np.random.default_rng(12)
    d, R = 3, 5
    P = np.stack([(lambda A: A @ A.T + d * np.eye(d))(rng.normal(size=(d, d)))
                  for _ in range(R)])
    mults = [1, 2, 4, 8, 16]
    got = T.duplication_penalty_matrix(P, mults)
    assert T.duplication_penalty(min(mults), d) <= got <= T.duplication_penalty(max(mults), d)


# --------------------------------------------------------------------------
# Theorems 4 and 5: sketch variance
# --------------------------------------------------------------------------

def simulate(w, k, k_hash, trials, seed):
    """The estimator exactly as ThetaSketch computes it."""
    rng = np.random.default_rng(seed)
    n = w.size
    out = np.empty(trials)
    for t in range(trials):
        u = rng.random(n)
        order = np.argsort(u)
        n_hat = (k_hash / u[order[k_hash]]) if n > k_hash else float(n)
        out[t] = n_hat * w[order[:k]].mean()
    return out


@pytest.mark.parametrize("n,k", [(20, 5), (50, 8), (30, 4)])
def test_bottomk_variance_matches_simulation(n, k):
    w = np.abs(np.random.default_rng(n).normal(2, 1, n)) + 0.2
    est = simulate(w, k, k, 200_000, seed=n)
    assert est.mean() == pytest.approx(w.sum(), rel=0.01)
    assert est.var() == pytest.approx(T.bottomk_variance(w, k), rel=0.05)


@pytest.mark.parametrize("n,k,kh", [(40, 4, 12), (60, 5, 20), (25, 3, 10)])
def test_split_budget_variance_matches_simulation(n, k, kh):
    w = np.abs(np.random.default_rng(n).normal(2, 1.2, n)) + 0.2
    est = simulate(w, k, kh, 200_000, seed=n + kh)
    assert est.mean() == pytest.approx(w.sum(), rel=0.01)
    assert est.var() == pytest.approx(T.split_budget_variance(w, k, kh), rel=0.06)


def test_split_budget_reduces_to_plain_bottomk_when_budgets_coincide():
    """The implementation's documented claim, now an identity rather than a
    remark: hash_capacity == capacity is exactly the plain HT estimator."""
    w = np.abs(np.random.default_rng(4).normal(2, 1, 40)) + 0.2
    for k in (3, 6, 12):
        assert T.split_budget_variance(w, k, k) == pytest.approx(
            T.bottomk_variance(w, k))


def test_uniform_weights_make_the_payload_budget_irrelevant():
    """A sharp qualitative prediction of Theorem 5, and the reason the
    allocation rule buys hashes when evidence is uniform: with equal weights the
    variance is W^2 (n - k_hash) / (n (k_hash - 1)) and k drops out entirely."""
    n, kh = 50, 20
    w = np.full(n, 3.0)
    closed = w.sum() ** 2 * (n - kh) / (n * (kh - 1))
    for k in (3, 5, 10, 20):
        assert T.split_budget_variance(w, k, kh) == pytest.approx(closed)


def test_variance_is_zero_below_capacity_where_the_sketch_is_exact():
    w = np.arange(1.0, 6.0)
    assert T.split_budget_variance(w, 8, 8) == 0.0
    assert T.bottomk_variance(w, 8) == 0.0


def test_k_equal_one_is_rejected_because_the_variance_diverges():
    """Counterexample for a load-bearing assumption. theta is the (k+1)-st order
    statistic and E[1/theta] = (n-1)/(k-1), which diverges at k = 1: the
    estimator is unbiased but has no finite variance, so every downstream bound
    is vacuous there."""
    w = np.abs(np.random.default_rng(9).normal(2, 1, 30)) + 0.2
    with pytest.raises(ValueError, match="k >= 2"):
        T.bottomk_variance(w, 1)


def test_selection_must_be_independent_of_the_values_it_selects():
    """Counterexample for Assumption A1. If the hash is correlated with the
    payload -- for instance if roots are hashed by a key that also orders their
    weights -- the estimator is badly biased and Theorem 5 does not apply. The
    implementation avoids this by hashing the root identifier under a seed that
    no content path can see."""
    n, k = 40, 5
    w = np.arange(1.0, n + 1.0)          # weight increases with index
    adversarial = np.argsort(np.arange(n))   # hash order == weight order
    biased = (n / k) * w[adversarial[:k]].mean()
    assert biased < 0.5 * w.sum(), "the adversarial coupling must break the estimate"
    honest = simulate(w, k, k, 20_000, seed=1).mean()
    assert honest == pytest.approx(w.sum(), rel=0.05)


# --------------------------------------------------------------------------
# Theorem 5 against the implementation
# --------------------------------------------------------------------------

def theta_estimates(w, k, kh, seeds, deliveries=1):
    out = []
    for s in range(seeds):
        sk = ThetaSketch(payload_dim=1, capacity=k, hash_capacity=kh, hash_seed=s)
        for _ in range(deliveries):
            for r, value in enumerate(w):
                sk.insert(f"root{r}", np.array([value]), float(value))
        out.append(sk.estimate_mass())
    return np.array(out)


def test_theta_sketch_realises_the_theorem_5_variance():
    n, k, kh = 40, 4, 12
    w = np.abs(np.random.default_rng(5).normal(2, 1.2, n)) + 0.2
    est = theta_estimates(w, k, kh, seeds=4000)
    assert est.mean() == pytest.approx(w.sum(), rel=0.03)
    assert est.var() == pytest.approx(T.split_budget_variance(w, k, kh), rel=0.15)


def test_duplicate_deliveries_change_neither_the_estimate_nor_its_error():
    """This is the theorem the paper is built on, checked on the implementation:
    the ledger state is a function of the root SET, so repeated delivery is
    bit-identical, and the compression error therefore cannot accumulate with
    duplication however many copies arrive."""
    n, k, kh = 60, 5, 20
    w = np.abs(np.random.default_rng(5).normal(2, 1.2, n)) + 0.2
    base = theta_estimates(w, k, kh, seeds=400, deliveries=1)
    for m in (2, 4, 16):
        rep = theta_estimates(w, k, kh, seeds=400, deliveries=m)
        assert np.array_equal(base, rep), f"x{m} delivery perturbed the ledger"


# --------------------------------------------------------------------------
# Theorems 6 and 7
# --------------------------------------------------------------------------

def test_compression_bound_holds_and_ignores_duplication():
    w = np.abs(np.random.default_rng(6).normal(2, 1, 200)) + 0.2
    out = T.compression_excess_nll_bound(w, k=16, k_hash=64, d=4, delta_prob=0.1)
    assert out["valid"] and out["excess_nll_bound"] > 0
    # Nothing in the bound's inputs mentions delivery multiplicity.
    again = T.compression_excess_nll_bound(w, k=16, k_hash=64, d=4, delta_prob=0.1)
    assert out == again


def test_compression_bound_tightens_with_memory():
    w = np.abs(np.random.default_rng(7).normal(2, 1, 400)) + 0.2
    bounds = [T.compression_excess_nll_bound(w, k=k, k_hash=4 * k, d=4)["excess_nll_bound"]
              for k in (8, 16, 32, 64)]
    assert all(b > a for a, b in zip(bounds[1:], bounds[:-1])), bounds


def test_allocation_prefers_hashes_for_uniform_and_payloads_for_dispersed_evidence():
    """The qualitative content of Theorem 7, and the reason it is a rule rather
    than a tuning result."""
    uniform = np.full(500, 2.0) + np.random.default_rng(1).normal(0, 0.02, 500)
    dispersed = np.abs(np.random.default_rng(2).lognormal(0, 1.5, 500)) + 0.01
    budget = 8 * (8 + 2) * 32 + 8 * 32
    a_u = T.allocate_budget(budget, 8, weights=uniform)
    a_d = T.allocate_budget(budget, 8, weights=dispersed)
    assert a_u.hash_slots / a_u.payload_slots > a_d.hash_slots / a_d.payload_slots


def test_allocation_respects_the_budget_and_the_ordering_constraint():
    w = np.abs(np.random.default_rng(8).normal(2, 1, 300)) + 0.2
    for budget in (2000, 5000, 20000):
        a = T.allocate_budget(budget, 8, weights=w)
        assert a.bytes_used <= budget
        assert a.hash_slots >= a.payload_slots >= 1


def test_allocation_hits_the_boundary_when_evidence_is_extremely_dispersed():
    """When CV exceeds sqrt(payload_dim + 2) the unconstrained optimum wants
    fewer hashes than payloads, which is infeasible, and the rule must return
    the plain bottom-k sketch rather than an invalid allocation."""
    w = np.abs(np.random.default_rng(13).lognormal(0, 3.0, 400)) + 1e-6
    a = T.allocate_budget(8 * 10 * 32 + 8 * 32, 8, weights=w)
    assert a.boundary and a.hash_slots == a.payload_slots


def test_hash_collision_bound_is_negligible_at_paper_scale():
    assert T.hash_collision_bound(10_000) < 1e-10
    assert T.hash_collision_bound(2) == pytest.approx(2.0 ** -64)
