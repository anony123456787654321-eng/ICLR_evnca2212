"""Label-aware scoring of recovered sector structure.

An arbitrary divergence threshold cannot say whether sectors are the RIGHT
sectors.  The ambiguity benchmark knows which hypothesis each cell's evidence
supports, so recovery is scored against those labels with the standard
clustering measures: adjusted Rand index and normalised mutual information,
both chance-corrected, so a model that puts every cell in one sector -- or
scatters them at random -- scores ~0 rather than being flattered.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import comb


def _contingency(labels_true, labels_pred):
    tu, ti = np.unique(labels_true, return_inverse=True)
    pu, pi = np.unique(labels_pred, return_inverse=True)
    C = np.zeros((len(tu), len(pu)), dtype=np.int64)
    np.add.at(C, (ti, pi), 1)
    return C


def adjusted_rand_index(labels_true, labels_pred) -> float:
    C = _contingency(np.asarray(labels_true), np.asarray(labels_pred))
    n = C.sum()
    if n < 2:
        return 0.0
    sum_ij = comb(C, 2).sum()
    sum_i = comb(C.sum(axis=1), 2).sum()
    sum_j = comb(C.sum(axis=0), 2).sum()
    expected = sum_i * sum_j / comb(n, 2)
    maximum = 0.5 * (sum_i + sum_j)
    denom = maximum - expected
    return float((sum_ij - expected) / denom) if abs(denom) > 1e-12 else 0.0


def normalised_mutual_info(labels_true, labels_pred) -> float:
    C = _contingency(np.asarray(labels_true), np.asarray(labels_pred)).astype(float)
    n = C.sum()
    if n == 0:
        return 0.0
    pij = C / n
    pi, pj = pij.sum(axis=1, keepdims=True), pij.sum(axis=0, keepdims=True)
    nz = pij > 0
    mi = float((pij[nz] * np.log(pij[nz] / (pi @ pj)[nz])).sum())
    hi = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    hj = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    denom = 0.5 * (hi + hj)
    return float(mi / denom) if denom > 1e-12 else 0.0


def score_sectors(hard: torch.Tensor, cell_label: torch.Tensor):
    """Mean ARI / NMI between hard sector assignment and true membership."""
    hard = hard.detach().cpu().numpy()
    lab = cell_label.detach().cpu().numpy()
    ari = [adjusted_rand_index(lab[b], hard[b]) for b in range(hard.shape[0])]
    nmi = [normalised_mutual_info(lab[b], hard[b]) for b in range(hard.shape[0])]
    return float(np.mean(ari)), float(np.mean(nmi))
