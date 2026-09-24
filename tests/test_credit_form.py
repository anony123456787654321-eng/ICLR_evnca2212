"""The scalar credit forms keep total information and drop only direction."""
import numpy as np
import torch

from ecnca.neural.dataset import BatchSpec, make_batch
from ecnca.neural.mpnn import MPNNEvidence


def _outputs(form, obs_dim=1, seed=0):
    spec = BatchSpec(n_cells=8, dim=4, obs_dim=obs_dim, max_roots=8,
                     max_occurrences=96, topology="path")
    batch = make_batch(spec, 4, 3, rng=np.random.default_rng(seed))
    torch.manual_seed(0)
    model = MPNNEvidence(dim=4, obs_dim=obs_dim, hidden=16,
                         variant="mpnn_ec", credit_form=form).eval()
    with torch.no_grad():
        return model(batch, steps=4)


def test_scalar_form_matches_trace_and_is_isotropic():
    m, s = _outputs("matrix"), _outputs("scalar")
    # Same total credited information above the prior.
    assert torch.allclose(m["M"], s["M"], atol=1e-5)
    lam = s["Lam"]
    eye = torch.eye(4)
    iso = lam.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
    assert torch.allclose(lam, iso, atol=1e-5)


def test_matrix_form_is_rank_deficient_under_rank_one_observations():
    lam = _outputs("matrix", obs_dim=1)["Lam"]
    eig = torch.linalg.eigvalsh(lam)
    # Three rank-one roots in four dimensions leave a direction at the prior.
    assert torch.isclose(eig[..., 0].min(), torch.tensor(1.0), atol=1e-4)


def test_fitted_scale_starts_at_the_fixed_scalar_form():
    a, b = _outputs("scalar"), _outputs("scalar_fit")
    assert torch.allclose(a["Lam"], b["Lam"], atol=1e-6)
    model = MPNNEvidence(dim=4, obs_dim=1, hidden=16, credit_form="scalar_fit")
    assert any(n == "log_scale" for n, _ in model.named_parameters())
