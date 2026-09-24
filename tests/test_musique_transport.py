import copy

import torch

from ecnca.real.encode import EmbeddingCache, HashEncoder
from ecnca.real.musique import parse_record
from ecnca.real.musique_transport import (MuSiQueTransport, make_transport_batch)
from tests.test_musique import fixture


def record():
    return parse_record(fixture())


def test_matched_variants_have_identical_parameters(tmp_path):
    counts = {name: sum(p.numel() for p in MuSiQueTransport(32, 16, name).parameters())
              for name in ("full", "filter_only", "no_provenance", "plain")}
    assert len(set(counts.values())) == 1


def test_downstream_private_input_cannot_change_upstream_message(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "cache.npz"), HashEncoder(32))
    original = record()
    changed = copy.deepcopy(fixture())
    changed["paragraphs"][1]["paragraph_text"] = "Entirely different downstream evidence."
    changed = parse_record(changed)
    model = MuSiQueTransport(32, 16, "full").eval()
    first = model(make_transport_batch([original], cache), collect=True)["versions"][:, 0]
    second = model(make_transport_batch([changed], cache), collect=True)["versions"][:, 0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_filter_payload_cannot_receive_downstream_refinement(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "cache.npz"), HashEncoder(32))
    model = MuSiQueTransport(32, 16, "filter_only").eval()
    output = model(make_transport_batch([record()], cache), collect=True)
    terminal, credit = model.ledger_readout(output["versions"], "filter_only", [0, 1])
    torch.testing.assert_close(terminal, output["versions"][:, 0], rtol=0, atol=0)
    assert credit == 1.0


def test_exact_redelivery_and_cycles_preserve_latest_full_payload(tmp_path):
    cache = EmbeddingCache(str(tmp_path / "cache.npz"), HashEncoder(32))
    model = MuSiQueTransport(32, 16, "full").eval()
    versions = model(make_transport_batch([record()], cache), collect=True)["versions"]
    copies, credit = model.ledger_readout(versions, "full", [1] * 16)
    cycle, cycle_credit = model.ledger_readout(versions, "full", [0, 1, 0, 1] * 4)
    torch.testing.assert_close(copies, versions[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(cycle, versions[:, 1], rtol=0, atol=0)
    assert credit == cycle_credit == 1.0
