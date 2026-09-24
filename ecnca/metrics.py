"""Scoring against the centralised unique-evidence posterior."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .gaussian import GaussianSpec, coverage_95, gaussian_nll, posterior, unpack


def confidence(spec: GaussianSpec, payload_sum: np.ndarray,
               payload_cov: np.ndarray | None = None) -> float:
    """Information content of a belief, in nats: 0.5 * log det Lambda_post.

    Computed from eigenvalues rather than ``slogdet`` of the assembled matrix.
    Methods that inflate Lambda to ~1e24 while the prior still contributes 1
    produce a condition number past float64 resolution, and ``slogdet`` then
    reports sign 0 -- which silently reads as "-inf confidence", i.e. the most
    overconfident methods would score as the most humble ones.
    """
    _, _, Lam = posterior(spec, payload_sum, payload_cov)
    w = np.linalg.eigvalsh(0.5 * (Lam + Lam.T))
    return float(0.5 * np.log(np.clip(w, 1e-300, None)).sum())


def evidence_mass(spec: GaussianSpec, payload_sum: np.ndarray,
                  payload_cov: np.ndarray | None = None) -> float:
    """Total reported information, tr(Lambda_post).

    Reported alongside ``confidence`` because log-det is rank-sensitive: with
    observations of rank < dim, a single root can never fill the space, so
    log-det would prefer two distinct roots for reasons of rank rather than of
    evidence.  Trace has no such artefact.
    """
    _, _, Lam = posterior(spec, payload_sum, payload_cov)
    return float(np.trace(Lam))


def score_belief(spec: GaussianSpec, payload_sum: np.ndarray, x_true: np.ndarray,
                 reference_sum: np.ndarray,
                 payload_cov: np.ndarray | None = None) -> Dict[str, float]:
    mean, Sigma, Lam = posterior(spec, payload_sum, payload_cov)
    ref_mean, ref_Sigma, ref_Lam = posterior(spec, reference_sum)
    return {
        "rmse": float(np.sqrt(np.mean((mean - x_true) ** 2))),
        "rmse_vs_oracle": float(np.sqrt(np.mean((mean - ref_mean) ** 2))),
        "nll": gaussian_nll(x_true, mean, Sigma),
        "nll_oracle": gaussian_nll(x_true, ref_mean, ref_Sigma),
        "coverage95": coverage_95(x_true, mean, Sigma),
        "prec_trace": float(np.trace(Lam)),
        "prec_trace_oracle": float(np.trace(ref_Lam)),
        "prec_rel_error": float(np.linalg.norm(Lam - ref_Lam) / max(np.linalg.norm(ref_Lam), 1e-12)),
        "prec_trace_ratio": float(np.trace(Lam) / max(np.trace(ref_Lam), 1e-12)),
        "confidence": confidence(spec, payload_sum, payload_cov),
        "confidence_oracle": confidence(spec, reference_sum),
    }


def score_cells(spec: GaussianSpec, payload_sums: List[np.ndarray], x_true: np.ndarray,
                reference_sum: np.ndarray,
                payload_covs: List[np.ndarray] | None = None) -> Dict[str, float]:
    """Mean over cells plus the consensus spread (max-min confidence)."""
    covs = payload_covs if payload_covs is not None else [None] * len(payload_sums)
    per_cell = [score_belief(spec, p, x_true, reference_sum, c)
                for p, c in zip(payload_sums, covs)]
    out = {k: float(np.mean([s[k] for s in per_cell])) for k in per_cell[0]}
    confs = [s["confidence"] for s in per_cell]
    traces = [s["prec_trace"] for s in per_cell]
    out["consensus_spread"] = float(np.max(confs) - np.min(confs))
    out["prec_trace_spread"] = float(np.max(traces) - np.min(traces))
    return out
