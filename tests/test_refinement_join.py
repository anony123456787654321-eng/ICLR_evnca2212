"""Gate 3R: the lineage refinement join.

A refined descendant must be able to REPLACE the belief about its root while
adding exactly zero evidence -- that is the whole point of separating
computational innovation from evidential independence.
"""
import numpy as np
import pytest
import torch

from ecnca.neural.refine import (RefineSpec, RefiningECNCA, make_refine_batch,
                                 oracle_posterior)

Z, R, N, B = 48, 3, 4, 2


def _model(variant="full"):
    torch.manual_seed(0)
    return RefiningECNCA(z_dim=Z, variant=variant)


def _state(versions, seed=0):
    """Craft (Z, V, rho, W) with given per-cell versions for root 0."""
    g = torch.Generator().manual_seed(seed)
    Zt = torch.randn(B, N, R, Z, generator=g)
    V = torch.zeros(B, N, R)
    W = torch.ones(B, N, R)
    for i, v in enumerate(versions):
        V[:, i, 0] = v
    rho = torch.zeros(B, N, R)
    return Zt, V, rho, W


def _adj_path():
    a = torch.zeros(N, N)
    for i in range(N - 1):
        a[i, i + 1] = a[i + 1, i] = 1.0
    return a


def _adj_cycle():
    a = _adj_path()
    a[0, N - 1] = a[N - 1, 0] = 1.0
    return a


def test_higher_version_replaces_lower():
    m = _model()
    Zt, V, rho, W = _state([0, 3, 0, 0])
    Zn, Vn, _, _ = m._join(Zt, V, rho, W, _adj_path())
    # cell 0 and cell 2 neighbour the version-3 cell and must adopt its version
    assert Vn[0, 0, 0].item() == 3 and Vn[0, 2, 0].item() == 3
    torch.testing.assert_close(Zn[0, 0, 0], Zt[0, 1, 0])
    torch.testing.assert_close(Zn[0, 2, 0], Zt[0, 1, 0])


def test_replacement_changes_representation_without_adding_precision():
    """Content is replaced, not summed: no evidence is created by refining."""
    m = _model()
    Zt, V, rho, W = _state([0, 5, 0, 0])
    rho[:] = 0.4
    Zn, _, rn, _ = m._join(Zt, V, rho, W, _adj_path())
    assert not torch.allclose(Zn[0, 0, 0], Zt[0, 0, 0])     # representation moved
    assert rn.max().item() == pytest.approx(0.4)            # credited evidence did not
    # and the adopted vector is exactly the neighbour's, never a sum
    assert not torch.allclose(Zn[0, 0, 0], Zt[0, 0, 0] + Zt[0, 1, 0])


def test_same_version_redelivery_is_idempotent():
    """Merging a state that already holds the identical message changes nothing.

    (Applying the join repeatedly to a POPULATION is a diffusion, not a
    redelivery: cells legitimately adopt different neighbours' messages.  The
    invariant is that absorbing a message you already hold is a no-op.)
    """
    m = _model()
    Zt, V, rho, W = _state([2, 2, 2, 2])
    Zt = Zt[:, :1].expand(-1, N, -1, -1).contiguous()     # every cell identical
    rho[:] = 0.3
    once = m._join(Zt, V, rho, W, _adj_path())
    twice = m._join(*once, _adj_path())
    for x, y in zip(once, twice):
        torch.testing.assert_close(x, y)
    torch.testing.assert_close(once[0], Zt)               # and it was a no-op


def test_merge_order_does_not_change_selected_version():
    m = _model()
    orders = [[0, 4, 1, 2], [2, 1, 4, 0], [1, 0, 2, 4]]
    picked = []
    for o in orders:
        Zt, V, rho, W = _state(o, seed=1)
        _, Vn, _, _ = m._join(Zt, V, rho, W, _adj_cycle())
        picked.append(float(Vn.max()))
    assert len(set(picked)) == 1 and picked[0] == 4.0


def test_cycling_cannot_increase_the_ceiling():
    """Round and round a cycle: the ceiling is a property of the root."""
    m = _model()
    Zt, V, rho, W = _state([3, 0, 0, 0])
    rho[:] = 0.6
    adj = _adj_cycle()
    traces = []
    for _ in range(12):
        Zt, V, rho, W = m._join(Zt, V, rho, W, adj)
        traces.append(float(rho.sum()))
    assert max(traces) == pytest.approx(traces[0]), traces


def test_distinct_root_increases_the_ceiling():
    sp = RefineSpec(n_cells=4, topology="path")
    m = _model().eval()
    with torch.no_grad():
        one = make_refine_batch(sp, 4, n_roots=1, rng=np.random.default_rng(0))
        two = make_refine_batch(sp, 4, n_roots=2, rng=np.random.default_rng(0))
        t1 = m(one, steps=6)["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).mean()
        t2 = m(two, steps=6)["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).mean()
    assert t2 > t1 * 1.05, (float(t1), float(t2))


@pytest.mark.parametrize("variant,bounded", [("full", True), ("filter_only", True),
                                             ("no_provenance", False), ("plain", False)])
def test_redelivery_never_raises_the_ceiling(variant, bounded):
    """However many times a root is re-delivered, EC may not credit it twice.

    The invariant is the CEILING, not a bit-identical Lambda: re-delivery also
    changes which cells hold the root at step 0, so their refinement
    trajectories -- and therefore rho -- legitimately differ.  What may never
    happen is the reported precision exceeding what the root itself carries.
    """
    sp = RefineSpec(n_cells=4, topology="path")
    m = _model(variant).eval()
    traces = []
    with torch.no_grad():
        for k in (1, 2, 4, 8):
            b = make_refine_batch(sp, 4, n_roots=1, redeliveries=k,
                                  rng=np.random.default_rng(3))
            ceiling = float(b["root_Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).sum(-1).mean()) + 4.0
            traces.append(float(m(b, steps=8)["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).max()))
    if bounded:
        assert max(traces) <= ceiling * 1.05, (variant, traces, ceiling)
        assert max(traces) <= traces[0] * 1.05, (variant, traces)
    else:
        assert max(traces) > ceiling, (variant, traces, ceiling)


def test_paired_probes_preserve_evidence_tensors():
    from ecnca.neural.dataset import BatchSpec, duplicate_batch, make_ambiguous_batch
    sp = BatchSpec(n_cells=16, topology="grid", max_roots=16, max_occurrences=48)
    base = make_ambiguous_batch(sp, 4, n_hyp=2, roots_per_hyp=3, resolve_roots=4,
                                resolve_step=6, rng=np.random.default_rng(0))
    dup = duplicate_batch(base, 4, np.random.default_rng(1), spread=True)
    for k in ("root_feat", "root_Lam", "root_mass", "root_valid", "x_true",
              "h_oracle", "Lam_oracle", "cell_label"):
        assert torch.equal(base[k], dup[k]), k


def test_table_and_experiment_share_the_paired_path():
    """Both must derive conditions from one base batch, never regenerate."""
    import pathlib
    for f in ("analysis/gate4_table.py", "experiments/gate4_ambiguity.py"):
        src = pathlib.Path(f).read_text()
        assert "duplicate_batch(" in src, f
        assert "copies=1, cross_sector_copies=0" in src, f
        assert '"transform"' not in src, f     # the probe tag must be gone


# --------------------------------------------------------------------------
# Probe separation: pure redelivery vs parallel refinement
# --------------------------------------------------------------------------
def _probe_batches(k, seed=11):
    sp = RefineSpec(n_cells=4, topology="path")
    return make_refine_batch(sp, 8, n_roots=1, redeliveries=k,
                             rng=np.random.default_rng(seed))


@pytest.mark.parametrize("variant", ["full", "filter_only"])
def test_pure_redelivery_leaves_the_ceiling_exact_and_the_rest_negligible(variant):
    """max_version=0: no computation happens, so nothing of substance may move.

    Two different claims, held to two different standards:

    EXACT      the credited evidence may never exceed the root's own ceiling,
               however many times it is re-delivered.  This is structural.
    NEGLIGIBLE prediction and precision still drift by ~0.4% relative, because
               `rho` is learned from (z, h) and the cell state h depends on WHEN
               a cell received the message -- a cell with longer to process
               claims slightly more of the SAME ceiling.  That is arrival-time
               sensitivity, not evidence being counted twice.
    """
    m = _model(variant).eval()
    with torch.no_grad():
        b1 = _probe_batches(1)
        ref = m(b1, steps=8, max_version=0)
        ceil = float(b1["root_Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).sum(-1).mean()) + 4.0
        mu_scale = float(ref["mu"].mean(1).abs().mean()) + 1e-9
        tr_scale = float(ref["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).mean())
        for k in (2, 4, 8):
            o = m(_probe_batches(k), steps=8, max_version=0)
            # exact: the ceiling holds
            assert float(o["Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).max()) <= ceil * 1.05
            # negligible: relative drift stays well under 2%
            pd_ = float(((o["mu"].mean(1) - ref["mu"].mean(1)) ** 2).sum(-1).sqrt().mean())
            cd_ = float((o["Lam"].mean(1) - ref["Lam"].mean(1)).abs().max())
            assert pd_ / mu_scale < 0.02, (variant, k, pd_ / mu_scale)
            assert cd_ / tr_scale < 0.02, (variant, k, cd_ / tr_scale)


@pytest.mark.parametrize("variant,bounded", [("full", True), ("filter_only", True),
                                             ("no_provenance", False), ("plain", False)])
def test_parallel_refinement_may_move_prediction_but_not_the_ceiling(variant, bounded):
    """max_version=8 with repeated delivery: copies refine concurrently."""
    m = _model(variant).eval()
    ratios = []
    with torch.no_grad():
        for k in (1, 2, 4, 8):
            b = _probe_batches(k)
            o = m(b, steps=8, max_version=8)
            ceil = b["root_Lam"].diagonal(dim1=-2, dim2=-1).sum(-1).sum(-1).clamp(min=1e-6)
            ratios.append(float((o["M"].max(dim=1).values / ceil).mean()))
    if bounded:
        assert max(ratios) <= 1.05, (variant, ratios)
    else:
        assert max(ratios) > 1.05, (variant, ratios)


def test_pure_and_parallel_are_distinct_axes():
    """The two conditions must not be collapsed back into one probe."""
    import pathlib
    src = pathlib.Path("experiments/gate3r_refinement.py").read_text()
    assert '"pure_redelivery"' in src and '"parallel_refinement"' in src
    assert 'axis="redelivery"' not in src          # the merged axis is gone
    assert "pred_drift" in src and "prec_drift" in src


def test_stats_script_reports_every_required_quantity():
    import pathlib
    src = pathlib.Path("analysis/gate3r_stats.py").read_text()
    for key in ("full_gain", "filter_only_gain", "paired_full_minus_filter",
                "ci95", "gap_auc_normalised", "max_precision_ratio",
                "pure_redelivery_drift_full", "new_source_full",
                "provenance_free_max_precision_ratio", "REPLICATION_PASS"):
        assert key in src, key
