"""Guards for the cost benchmark.

A cost measurement is easy to get wrong in a way that flatters whichever method
the author prefers, so these tests fix the two things that must be true: the
arrival accounting has to agree with the ledger that produced it, and the two
cost columns have to mean what the table says they mean.
"""
import importlib.util
from pathlib import Path

import pytest

from ecnca.real.musique import MuSiQueRecord, MuSiQueStep
from ecnca.real.musique_duplication import (base_stream, build_stream,
                                            ledger_mode, resolve_stream)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "cost_benchmark", ROOT / "experiments" / "cost_benchmark.py")
cost_benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cost_benchmark)


def record(n_hops=4, words=60):
    steps = tuple(MuSiQueStep(
        question=f"question {i}", answer=f"answer{i}",
        paragraph_title=f"Title {i}",
        paragraph_text=" ".join(f"w{i}x{j}" for j in range(words)),
        paragraph_idx=i, dependencies=() if i == 0 else (i,))
        for i in range(n_hops))
    return MuSiQueRecord(record_id="rec", question="Q", answer=f"answer{n_hops-1}",
                         answer_aliases=(), steps=steps, is_linear=True,
                         linear_failure="")


def measure(defence, intervention="exact", multiplicity=16, n=4):
    records = [record() for _ in range(n)]
    # Distinct record ids, so roots do not collide across the population.
    records = [MuSiQueRecord(record_id=f"rec{i}", question=r.question,
                             answer=r.answer, answer_aliases=r.answer_aliases,
                             steps=r.steps, is_linear=True, linear_failure="")
               for i, r in enumerate(records)]
    return records, cost_benchmark.ledger_cost(
        records, defence, intervention, multiplicity, "one_hop", 17, None, 2)


@pytest.mark.parametrize("defence", ["full", "filter_only", "plain",
                                     "canonical_dedup", "minhash_dedup",
                                     "simhash_dedup"])
def test_arrival_accounting_agrees_with_the_ledger(defence):
    """admitted + suppressed must equal arrivals, for every defence."""
    _, cost = measure(defence)
    assert cost["admitted"] + cost["suppressed"] == cost["arrivals"]
    assert cost["admitted"] >= 1


def test_conservation_admits_no_more_than_any_dedup_and_far_less_than_none():
    """The forward column only makes sense if this ordering holds: conservation
    admits at most what deduplication admits, and both admit far less than no
    defence."""
    _, ec = measure("full")
    _, canonical = measure("canonical_dedup")
    _, plain = measure("plain")
    assert ec["admitted"] <= canonical["admitted"] < plain["admitted"]
    assert plain["admitted"] == plain["arrivals"], (
        "the no-defence control must process every arrival, or it is not a "
        "control")


def test_credited_evidence_matches_a_direct_resolve():
    """The cost path and the evaluation path must agree on credit, otherwise the
    cost table and the results table describe different systems."""
    records, cost = measure("full")
    direct = sum(
        sum(resolve_stream(build_stream(r, "exact", 16, "one_hop", 17),
                           ledger_mode("full"))[1].values())
        for r in records)
    assert cost["credited_evidence"] == pytest.approx(direct)


def test_timing_fields_are_positive_and_min_does_not_exceed_mean():
    _, cost = measure("minhash_dedup")
    assert cost["ledger_seconds_min"] > 0
    assert cost["ledger_seconds_min"] <= cost["ledger_seconds_mean"] + 1e-12
    assert cost["ledger_us_per_arrival"] > 0


def test_the_benchmark_covers_every_defence_the_paper_compares():
    from ecnca.real.musique_duplication import DEDUP_VARIANTS
    assert set(DEDUP_VARIANTS) <= set(cost_benchmark.DEFENCES)
    for control in ("full", "filter_only", "plain"):
        assert control in cost_benchmark.DEFENCES
