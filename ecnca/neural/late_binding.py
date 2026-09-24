"""Late-binding associative retrieval for dynamic-sector evaluation.

The task isolates the information that sectorisation is meant to preserve.
Several cell populations hold different key--value bindings.  A query revealing
which key matters arrives only after those populations have communicated.  A
single pooled state loses the pairing: swapping values between keys leaves both
global means unchanged while changing the correct answer.  A partitioned state
can retain one binding per active region and answer after the query arrives.

This is deliberately a mechanism benchmark, not a claim that clustering alone
solves RAG.  It supplies a falsifiable positive control before we spend GPU time
on a learned, text-backed version of the same late-binding problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dynamic_sectors import GrowingSectorController, SectorConfig
from .sector_metrics import adjusted_rand_index, normalised_mutual_info


@dataclass
class LateBindingScenario:
    embedding: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    labels: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor
    target_label: int
    true_keys: torch.Tensor
    true_values: torch.Tensor


def make_late_binding_scenario(
    seed: int,
    n_groups: int,
    cells_per_group: int | Sequence[int] = 8,
    key_dim: int = 5,
    value_dim: int = 4,
    noise: float = 0.03,
    value_geometry_scale: float = 0.05,
    key_cosine: float = 0.0,
) -> LateBindingScenario:
    """Create an exchangeable set of noisy key--value populations.

    Keys are orthogonal but randomly rotated, so sector identities have no
    fixed meaning across examples.  Values are independent of keys.  The late
    query names one key but does not reveal its associated value.
    """
    if not 1 <= n_groups < key_dim:
        raise ValueError("n_groups must be in [1, key_dim - 1]")
    if not 0.0 <= key_cosine < 1.0:
        raise ValueError("key_cosine must be in [0, 1)")
    rng = np.random.default_rng(seed)
    qmat, _ = np.linalg.qr(rng.normal(size=(key_dim, key_dim)))
    canonical = np.zeros((n_groups, key_dim), dtype=np.float64)
    canonical[:, 0] = np.sqrt(key_cosine)
    for group in range(n_groups):
        canonical[group, group + 1] = np.sqrt(1.0 - key_cosine)
    true_keys = (canonical @ qmat).astype(np.float32)
    true_values = rng.normal(size=(n_groups, value_dim)).astype(np.float32)
    target_label = int(rng.integers(0, n_groups))

    if isinstance(cells_per_group, int):
        group_sizes = [cells_per_group] * n_groups
    else:
        group_sizes = list(cells_per_group)
        if len(group_sizes) != n_groups or min(group_sizes) < 1:
            raise ValueError("cells_per_group must give one positive size per group")

    keys, values, labels = [], [], []
    for group in range(n_groups):
        size = group_sizes[group]
        keys.append(true_keys[group] + noise * rng.normal(size=(size, key_dim)))
        values.append(true_values[group] + noise * rng.normal(size=(size, value_dim)))
        labels.extend([group] * size)
    keys = np.concatenate(keys).astype(np.float32)
    values = np.concatenate(values).astype(np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(labels))
    keys, values, labels = keys[order], values[order], labels[order]

    # Sector geometry is dominated by the semantic key.  A small value term
    # allows content evolution to move a cell near a boundary without letting
    # arbitrary payload magnitude define the partition.
    embedding = np.concatenate([keys, value_geometry_scale * values], axis=-1)
    return LateBindingScenario(
        embedding=F.normalize(torch.from_numpy(embedding), dim=-1),
        keys=torch.from_numpy(keys),
        values=torch.from_numpy(values),
        labels=torch.from_numpy(labels),
        query=torch.from_numpy(true_keys[target_label]),
        target=torch.from_numpy(true_values[target_label]),
        target_label=target_label,
        true_keys=torch.from_numpy(true_keys),
        true_values=torch.from_numpy(true_values),
    )


def _group_means(x: torch.Tensor, hard: torch.Tensor, active_slots) -> torch.Tensor:
    return torch.stack([x[hard == slot].mean(0) for slot in active_slots])


def dynamic_sector_readout(
    scenario: LateBindingScenario,
    steps: int = 12,
    cfg: SectorConfig | None = None,
) -> Dict[str, float]:
    """Answer the late query from bindings retained by inferred sectors."""
    cfg = cfg or SectorConfig(dim=scenario.embedding.shape[-1], k_max=8)
    ctrl = GrowingSectorController(cfg)
    qs, state = ctrl.run([scenario.embedding] * steps)
    hard = qs[-1].argmax(-1)
    slots = [int(x) for x in torch.unique(hard)]
    mean_keys = F.normalize(_group_means(scenario.keys, hard, slots), dim=-1)
    mean_values = _group_means(scenario.values, hard, slots)
    selected = int(torch.argmax(mean_keys @ F.normalize(scenario.query, dim=-1)))
    pred = mean_values[selected]
    return {
        "squared_error": float((pred - scenario.target).square().mean()),
        "n_active": float(state.n_active),
        "n_occupied": float(len(slots)),
        "count_error": float(abs(len(slots) - len(scenario.true_keys))),
        "ari": adjusted_rand_index(scenario.labels.numpy(), hard.numpy()),
        "nmi": normalised_mutual_info(scenario.labels.numpy(), hard.numpy()),
        "births": float(state.births),
        "merges": float(state.merges),
        "retirements": float(state.retirements),
    }


def pooled_readout(scenario: LateBindingScenario) -> Dict[str, float]:
    """Bayes-optimal exchangeable readout after the key--value pairing is lost."""
    pred = scenario.values.mean(0)
    return {"squared_error": float((pred - scenario.target).square().mean())}


def oracle_readout(scenario: LateBindingScenario) -> Dict[str, float]:
    pred = scenario.values[scenario.labels == scenario.target_label].mean(0)
    return {"squared_error": float((pred - scenario.target).square().mean())}


def pooled_collision(seed: int = 0) -> Tuple[LateBindingScenario, LateBindingScenario]:
    """Two inputs with identical pooled states but different correct answers.

    The second input swaps complete value populations between the two keys.
    For the same query, global mean(key) and mean(value) are unchanged, while
    the requested value changes.  Therefore no deterministic function of those
    pooled statistics can solve both inputs.
    """
    first = make_late_binding_scenario(seed, 2, noise=0.0)
    second = LateBindingScenario(
        embedding=first.embedding.clone(),
        keys=first.keys.clone(),
        values=first.values.clone(),
        labels=first.labels.clone(),
        query=first.query.clone(),
        target=first.target.clone(),
        target_label=first.target_label,
        true_keys=first.true_keys.clone(),
        true_values=first.true_values.flip(0).clone(),
    )
    swapped = 1 - first.labels
    second.values = first.true_values[swapped].clone()
    second.target = second.true_values[second.target_label].clone()
    # Geometry is keyed, not value-defined; it must remain identical under a
    # value permutation or the collision would leak through another channel.
    second.embedding = first.embedding.clone()
    return first, second


def lifecycle_counts(
    scenario: LateBindingScenario,
    settle_steps: int = 12,
    resolve_steps: int = 24,
    cfg: SectorConfig | None = None,
) -> Dict[str, float]:
    """Measure growth before a late query and contraction after resolution."""
    cfg = cfg or SectorConfig(dim=scenario.embedding.shape[-1], k_max=8)
    ctrl = GrowingSectorController(cfg)
    state = ctrl.init_state(scenario.embedding)
    q = None
    for _ in range(settle_steps):
        q, state = ctrl.step(scenario.embedding, state)
    before = state.n_active

    key = scenario.query.expand_as(scenario.keys)
    value = scenario.target.expand_as(scenario.values)
    resolved = F.normalize(torch.cat([key, 0.05 * value], dim=-1), dim=-1)
    start = scenario.embedding
    transition_steps = min(8, max(2, resolve_steps // 2))
    collapse_delay = None
    for step in range(resolve_steps):
        amount = min(1.0, step / max(transition_steps - 1, 1))
        z = F.normalize((1 - amount) * start + amount * resolved, dim=-1)
        q, state = ctrl.step(z, state)
        if collapse_delay is None and state.n_active == 1:
            collapse_delay = step
    return {
        "active_before": float(before),
        "active_after": float(state.n_active),
        "total_births": float(state.births),
        "total_merges": float(state.merges),
        "total_retirements": float(state.retirements),
        "collapse_delay": float(collapse_delay) if collapse_delay is not None else float("nan"),
    }
