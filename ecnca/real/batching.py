"""Turn ClaimRecords into EC-RAG batches, with the bypass controls built in.

`control` produces an ABLATED view of the same batch so that "is the model
actually using the evidence?" can be answered rather than assumed:

  none              the real batch
  claim_only        every passage masked out
  shuffled_evidence passages permuted across examples in the batch
  removed_evidence  a fraction of passages dropped
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .averitec import ClaimRecord
from .ecrag import LABELS, theta_keep_mask

CONTROLS = ("none", "claim_only", "shuffled_evidence", "removed_evidence")


def make_batch(records: Sequence[ClaimRecord], cache, max_passages: int = 32,
               theta_capacity: int = 8, control: str = "none",
               drop_fraction: float = 0.5, seed: int = 0,
               device: str = "cpu") -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    B = len(records)
    texts_claim = [r.claim for r in records]
    per: List[List] = []
    for r in records:
        ps = r.passages[:max_passages]
        if control == "removed_evidence" and ps:
            keep = [p for p in ps if rng.random() >= drop_fraction]
            ps = keep or ps[:1]
        per.append(ps)
    if control == "claim_only":
        per = [[] for _ in records]
    if control == "shuffled_evidence" and B > 1:
        order = rng.permutation(B)
        while np.any(order == np.arange(B)):          # ensure a real derangement
            order = rng.permutation(B)
        per = [per[i] for i in order]

    N = max(1, max((len(p) for p in per), default=1))
    dim = cache.dim or 64
    claim_emb = torch.as_tensor(cache.encode(texts_claim), device=device)
    pas = torch.zeros(B, N, dim, device=device)
    mask = torch.zeros(B, N, device=device)
    rel = torch.zeros(B, N, device=device)
    root_index = torch.zeros(B, N, dtype=torch.long, device=device)
    root_ids: List[List[str]] = []
    R = 1
    for b, ps in enumerate(per):
        if ps:
            emb = torch.as_tensor(cache.encode([p.text for p in ps]), device=device)
            pas[b, :len(ps)] = emb
            mask[b, :len(ps)] = 1.0
            rel[b, :len(ps)] = torch.as_tensor([p.retrieval_score for p in ps],
                                               dtype=torch.float32, device=device)
        uniq = sorted({p.root_id for p in ps if p.root_id})
        slot = {r: i for i, r in enumerate(uniq)}
        for i, p in enumerate(ps):
            root_index[b, i] = slot.get(p.root_id, 0)
        root_ids.append([p.root_id for p in ps])
        R = max(R, len(uniq))
    first_of_root = torch.zeros(B, N, device=device)
    for b, ps in enumerate(per):
        seen = set()
        for i, pp in enumerate(ps):
            if pp.root_id not in seen:
                seen.add(pp.root_id)
                first_of_root[b, i] = 1.0
    root_mask = torch.zeros(B, R, device=device)
    for b, ps in enumerate(per):
        n = len({p.root_id for p in ps if p.root_id})
        root_mask[b, :n] = 1.0

    labels = torch.tensor([LABELS.index(r.label) if r.label in LABELS else -100
                           for r in records], device=device)
    keep = theta_keep_mask(root_ids, theta_capacity)
    if keep.shape[1] < N:
        keep = torch.cat([keep, torch.zeros(B, N - keep.shape[1])], dim=1)
    return {"claim_emb": claim_emb, "passage_emb": pas, "passage_mask": mask,
            "retrieval_score": rel, "root_index": root_index,
            "root_mask": root_mask, "labels": labels,
            "theta_keep": keep[:, :N].to(device), "first_of_root": first_of_root,
            "n_roots": root_mask.sum(-1), "control": control}
