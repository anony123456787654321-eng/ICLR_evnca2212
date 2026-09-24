"""EC-Gaussian: exact ground truth for evidence conservation.

Latent x ~ N(0, tau^-1 I_d).  Each *root* observation s is conditionally
independent given x:

    y_s = A_s x + eps_s,      eps_s ~ N(0, sigma_s^2 I_m)

Its natural-parameter contribution is additive over distinct roots:

    Lambda_s = A_s^T Sigma_s^-1 A_s,     h_s = A_s^T Sigma_s^-1 y_s
    Lambda_post = Lambda_prior + sum_{s in unique(E)} Lambda_s
    h_post      = h_prior      + sum_{s in unique(E)} h_s

Because the sum runs over *unique* roots, the centralised posterior below is the
exact target any evidence-conserving scheme must hit -- no matter how many
copies of a root the graph delivers, or how many times a message goes round a
cycle.  Everything a cell ships is the packed (h, vech(Lambda)) payload, so
merging is a set union of additive terms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def payload_dim(d: int) -> int:
    return d + d * (d + 1) // 2


def _tril_idx(d: int):
    return np.tril_indices(d)


def pack(h: np.ndarray, Lam: np.ndarray) -> np.ndarray:
    """(h, Lambda) -> flat additive payload (Lambda stored as vech)."""
    i, j = _tril_idx(Lam.shape[0])
    return np.concatenate([np.asarray(h, np.float64).ravel(), Lam[i, j]])


def unpack(vec: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    vec = np.asarray(vec, np.float64)
    h = vec[:d].copy()
    Lam = np.zeros((d, d))
    i, j = _tril_idx(d)
    Lam[i, j] = vec[d:]
    Lam = Lam + np.tril(Lam, -1).T
    return h, Lam


@dataclass
class GaussianSpec:
    dim: int = 4
    obs_dim: int = 2
    prior_precision: float = 1.0
    noise_sigma: float = 1.0

    @property
    def payload_dim(self) -> int:
        return payload_dim(self.dim)

    def prior(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self.dim), self.prior_precision * np.eye(self.dim)


def make_observation(spec: GaussianSpec, x: np.ndarray, rng: np.random.Generator):
    """Draw one conditionally independent observation and return its payload."""
    A = rng.normal(0.0, 1.0, size=(spec.obs_dim, spec.dim)) / np.sqrt(spec.dim)
    sigma = spec.noise_sigma
    y = A @ x + rng.normal(0.0, sigma, size=spec.obs_dim)
    prec = 1.0 / (sigma ** 2)
    Lam = prec * (A.T @ A)
    h = prec * (A.T @ y)
    return pack(h, Lam), float(np.trace(Lam))


def posterior(spec: GaussianSpec, payload_sum: np.ndarray,
              payload_cov: np.ndarray | None = None):
    """Natural-parameter sum -> (mean, covariance, precision).

    ``payload_cov`` is the sampling covariance of the payload estimate itself
    (zero for an exact ledger).  A cell that compressed its ledger knows the
    magnitude of the evidence but located the mean from a subsample, so the
    reported covariance must carry that through:

        Sigma_report = Lambda^-1  +  Lambda^-1 V_h Lambda^-1

    Without this term the cell states n-root confidence around a k-root mean,
    which is the very error evidence conservation is meant to prevent.
    """
    h0, Lam0 = spec.prior()
    h, Lam = unpack(payload_sum, spec.dim)
    Lam_post = Lam0 + Lam
    h_post = h0 + h
    # Lambda can lose PSD-ness under a biased estimator; project for scoring.
    w, V = np.linalg.eigh(Lam_post)
    w = np.clip(w, 1e-8, None)
    Lam_post = (V * w) @ V.T
    Sigma = (V / w) @ V.T
    mean = Sigma @ h_post
    if payload_cov is not None and np.any(payload_cov):
        Vh = np.asarray(payload_cov)[:spec.dim, :spec.dim]
        Sigma = Sigma + Sigma @ Vh @ Sigma
        w2, V2 = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
        w2 = np.clip(w2, 1e-12, None)
        Sigma = (V2 * w2) @ V2.T
        Lam_post = (V2 / w2) @ V2.T
    return mean, Sigma, Lam_post


def gaussian_nll(x_true: np.ndarray, mean: np.ndarray, Sigma: np.ndarray) -> float:
    d = len(x_true)
    err = x_true - mean
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        logdet = np.log(np.clip(np.linalg.eigvalsh(Sigma), 1e-12, None)).sum()
    return float(0.5 * (err @ np.linalg.solve(Sigma, err) + logdet + d * np.log(2 * np.pi)))


def coverage_95(x_true: np.ndarray, mean: np.ndarray, Sigma: np.ndarray) -> float:
    """Fraction of coordinates inside the marginal 95% interval."""
    sd = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    return float(np.mean(np.abs(x_true - mean) <= 1.959963985 * sd))
