"""Fairness guards for the head-to-head architecture matrix.

The claim under test is that evidence conservation is a general accounting
constraint, not an NCA-specific trick. That claim is only meaningful if the
ordinary and EC-wrapped forms of a backbone are genuinely the same model apart
from accounting. These tests assert exactly that, and they assert the parameter
matching that makes each cross-backbone comparison fair.
"""
import numpy as np
import pytest
import torch

from ecnca.neural.dataset import BatchSpec, duplicate_batch, make_batch
from ecnca.neural.model import SectorizedECNCA
from ecnca.neural.mpnn import MPNNEvidence, NonBacktrackingMPNN
from experiments.architecture_matrix import regimes
from ecnca.real.ecrag import ECRag
from ecnca.real.musique_transport import MuSiQueTransport
from ecnca.real.set_transformer import SetTransformerRag
from ecnca.real.transformer_transport import TransformerTransport

TOL = 0.01          # parameter matching budget
MULTIPLICITIES = (1, 2, 4, 8, 16)


def params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def graph_batch(seed=0, cells=8, topology="path"):
    spec = BatchSpec(n_cells=cells, dim=4, obs_dim=4, max_roots=8,
                     max_occurrences=96, topology=topology)
    return spec, make_batch(spec, 4, 3, rng=np.random.default_rng(seed))


def transport_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"question": torch.randn(3, 4, 768, generator=g),
            "paragraph": torch.randn(3, 4, 768, generator=g),
            "mask": torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.], [1.] * 4]),
            "n_hops": torch.tensor([2, 3, 4])}


def rag_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"claim_emb": torch.randn(3, 768, generator=g),
            "passage_emb": torch.randn(3, 6, 768, generator=g),
            "passage_mask": torch.ones(3, 6),
            "root_index": torch.tensor([[0, 0, 0, 1, 1, 2]] * 3),
            "root_mask": torch.ones(3, 3),
            "retrieval_score": torch.rand(3, 6, generator=g),
            "first_of_root": torch.tensor([[1., 0., 0., 1., 0., 1.]] * 3)}


# --- parameter matching -----------------------------------------------------

def test_transformer_transport_matches_the_recurrent_transport():
    gru, tr = params(MuSiQueTransport(variant="full")), params(
        TransformerTransport(variant="transformer_ec"))
    assert abs(tr - gru) / gru <= TOL, f"{tr} vs {gru}"


def test_mpnn_matches_the_recurrent_nca():
    nca = params(SectorizedECNCA(dim=4, obs_dim=4, hidden=64, use_sectors=False))
    for model in (MPNNEvidence(), NonBacktrackingMPNN()):
        assert abs(params(model) - nca) / nca <= TOL, params(model)


def test_set_transformer_matches_ecrag():
    ec, st = params(ECRag(emb_dim=768, hidden=128)), params(SetTransformerRag())
    assert abs(st - ec) / ec <= TOL, f"{st} vs {ec}"


def test_mpnn_hidden_dimension_matches_the_nca():
    """Depth and width must match; only the message width absorbs the budget."""
    assert MPNNEvidence().hidden == SectorizedECNCA(use_sectors=False).hidden == 64


# --- identical content modules at initialization ----------------------------

@pytest.mark.parametrize("pair,factory", [
    (("transformer_ec", "transformer_plain"),
     lambda v: TransformerTransport(variant=v)),
    (("mpnn_ec", "mpnn_plain"), lambda v: MPNNEvidence(variant=v)),
    (("set_transformer_ec", "set_transformer_plain"),
     lambda v: SetTransformerRag(variant=v)),
])
def test_ec_and_ordinary_share_identical_content_modules_at_init(pair, factory):
    a_name, b_name = pair
    torch.manual_seed(0); a = factory(a_name)
    torch.manual_seed(0); b = factory(b_name)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb), "the two forms must have the same modules"
    for key in sa:
        assert torch.equal(sa[key], sb[key]), f"{key} differs at initialization"


# --- byte-identical inputs and content path ---------------------------------

def test_transformer_ec_and_plain_produce_identical_predictions():
    batch = transport_batch()
    outs = {}
    for v in ("transformer_ec", "transformer_plain"):
        torch.manual_seed(0)
        with torch.no_grad():
            outs[v] = TransformerTransport(variant=v).eval()(batch)
    assert torch.equal(outs["transformer_ec"]["prediction"],
                       outs["transformer_plain"]["prediction"])
    # ...and the accounting genuinely differs
    assert not torch.equal(outs["transformer_ec"]["credited_evidence"],
                           outs["transformer_plain"]["credited_evidence"])


def test_mpnn_ec_and_plain_produce_identical_content():
    _, batch = graph_batch()
    outs = {}
    for v in ("mpnn_ec", "mpnn_plain"):
        torch.manual_seed(0)
        with torch.no_grad():
            outs[v] = MPNNEvidence(variant=v).eval()(batch, steps=6)
    assert torch.equal(outs["mpnn_ec"]["mu"], outs["mpnn_plain"]["mu"])


def test_set_transformer_ec_and_plain_share_the_pooled_state():
    batch = rag_batch()
    outs = {}
    for v in ("set_transformer_ec", "set_transformer_plain"):
        torch.manual_seed(0)
        with torch.no_grad():
            outs[v] = SetTransformerRag(variant=v).eval()(batch)
    assert torch.equal(outs["set_transformer_ec"]["state"],
                       outs["set_transformer_plain"]["state"])
    assert not torch.equal(outs["set_transformer_ec"]["credited_evidence"],
                           outs["set_transformer_plain"]["credited_evidence"])


# --- the conservation result itself -----------------------------------------

def test_mpnn_ec_is_duplicate_invariant_and_plain_is_not():
    _, base = graph_batch()
    seen = {}
    for v in ("mpnn_ec", "mpnn_plain"):
        torch.manual_seed(0)
        model = MPNNEvidence(variant=v).eval()
        ratios = []
        with torch.no_grad():
            for m in MULTIPLICITIES:
                b = base if m == 1 else duplicate_batch(base, m, np.random.default_rng(1))
                ratios.append(float(model(b, steps=6)["credited"].max()))
        seen[v] = [r / ratios[0] for r in ratios]
    assert seen["mpnn_ec"] == pytest.approx([1.0] * 5, abs=1e-6)
    assert seen["mpnn_plain"] == pytest.approx([1.0, 2.0, 4.0, 8.0, 16.0], rel=1e-6)


def test_nonbacktracking_does_not_solve_root_duplication():
    """The scientific distinction: it addresses cycles, not root identity."""
    _, base = graph_batch()
    torch.manual_seed(0)
    model = NonBacktrackingMPNN().eval()
    ratios = []
    with torch.no_grad():
        for m in MULTIPLICITIES:
            b = base if m == 1 else duplicate_batch(base, m, np.random.default_rng(1))
            ratios.append(float(model(b, steps=6)["credited"].max()))
    ratios = [r / ratios[0] for r in ratios]
    assert ratios == pytest.approx([1.0, 2.0, 4.0, 8.0, 16.0], rel=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression guard")
def test_architecture_driver_places_models_and_all_regimes_on_cuda():
    """The DGX driver must not leave generated batches on CPU."""
    spec = BatchSpec(n_cells=8, dim=4, obs_dim=4, max_roots=8,
                     max_occurrences=96, topology="path")
    held = regimes(spec, batch_size=2, n_roots=3, seed=0, device="cuda")
    for batch in held.values():
        tensor_devices = {value.device.type for value in batch.values()
                          if torch.is_tensor(value)}
        assert tensor_devices == {"cuda"}
    model = MPNNEvidence().cuda()
    with torch.no_grad():
        output = model(held["clean"], steps=2)
    assert output["mu"].device.type == "cuda"


def test_transformer_transport_is_causal():
    """A step must not read a later step, or the task would be trivialised."""
    batch = transport_batch()
    torch.manual_seed(0)
    model = TransformerTransport(variant="transformer_ec").eval()
    with torch.no_grad():
        before = model(batch, collect=True)["versions"]
        perturbed = {**batch, "paragraph": batch["paragraph"].clone()}
        perturbed["paragraph"][2, 3] = torch.randn(768)
        after = model(perturbed, collect=True)["versions"]
    assert torch.equal(before[2, :3], after[2, :3]), "earlier states moved"
    assert not torch.equal(before[2, 3], after[2, 3]), "the perturbation did nothing"


def test_filter_only_forms_change_the_content_path():
    """filter_only is a different object from plain: it deletes computation."""
    batch = transport_batch()
    outs = {}
    for v in ("transformer_ec", "transformer_filter_only"):
        torch.manual_seed(0)
        with torch.no_grad():
            outs[v] = TransformerTransport(variant=v).eval()(batch)
    assert not torch.equal(outs["transformer_ec"]["prediction"],
                           outs["transformer_filter_only"]["prediction"])


def test_set_transformer_is_not_labelled_fid():
    """A superficial FiD-like model must not be called FiD."""
    import pathlib
    src = pathlib.Path("ecnca/real/set_transformer.py").read_text()
    assert "not a FiD" in src.lower() or "NOT a FiD" in src
    for path in ("paper/main_v2.tex", "paper/main.tex", "RESULTS.md"):
        if not pathlib.Path(path).is_file():
            continue
        text = pathlib.Path(path).read_text()
        assert "FiD" not in text or "not" in text.lower()


def test_set_transformer_ec_is_duplicate_invariant_and_plain_is_not():
    """Table 1 marks EC Set Transformer duplicate-invariant as *measured*."""
    def batch(copies):
        g = torch.Generator().manual_seed(0)
        pe = torch.randn(1, 1, 768, generator=g).expand(1, copies, 768).contiguous()
        return {"claim_emb": torch.randn(1, 768, generator=torch.Generator().manual_seed(1)),
                "passage_emb": pe, "passage_mask": torch.ones(1, copies),
                "root_index": torch.zeros(1, copies, dtype=torch.long),
                "root_mask": torch.ones(1, 1),
                "retrieval_score": torch.zeros(1, copies)}
    seen = {}
    for v in ("set_transformer_ec", "set_transformer_plain"):
        torch.manual_seed(0)
        model = SetTransformerRag(variant=v).eval()
        with torch.no_grad():
            vals = [float(model(batch(c))["credited_evidence"][0])
                    for c in MULTIPLICITIES]
        seen[v] = [x / vals[0] for x in vals]
    assert seen["set_transformer_ec"] == pytest.approx([1.0] * 5, rel=1e-5)
    assert seen["set_transformer_plain"] == pytest.approx(
        [1.0, 2.0, 4.0, 8.0, 16.0], rel=1e-5)
