"""Shortcut guards for the duplicated-hop MuSiQue evaluation.

A duplication benchmark is worthless if the intervention quietly changes the
task, so these tests assert that it does not: the label, the candidate pool and
the lineage-distinct root count are invariant, exact copies are byte-identical,
re-chunking and rewording never mint a new root, and nothing about multiplicity,
padding or sequence length is predictive of the answer.
"""
import random

import numpy as np
import pytest
import torch

from ecnca.real.musique import MuSiQueRecord, MuSiQueStep
from ecnca.real.musique_calibration import (Calibration, READOUTS,
                                            expected_calibration_error, fit,
                                            score)
from ecnca.real.musique_duplication import (DEDUP_VARIANTS, DEMPSTER_MASS,
                                            DETECTORS, FUSION_MODES,
                                            FUSION_VARIANTS,
                                            INTERVENTIONS, LEDGER_MODES,
                                            MULTIPLICITIES, SCOPES,
                                            base_stream, build_stream,
                                            dempster_credit, ledger_mode,
                                            overlapping_views,
                                            partial_windows,
                                            resolve_stream, stream_stats,
                                            version_key)

# The ledger modes of the original comparison. The near-duplicate detectors are
# a separate arm with its own guards in tests/test_near_duplicate.py, so they
# are named here only to keep this list exhaustive.
MODES = ("max", "first", "canonical", "none")


def record(n_hops=4, words=40):
    steps = tuple(MuSiQueStep(
        question=f"question {i}", answer=f"answer{i}",
        paragraph_title=f"Title {i}",
        paragraph_text=" ".join(f"w{i}x{j}" for j in range(words)),
        paragraph_idx=i, dependencies=() if i == 0 else (i,))
        for i in range(n_hops))
    return MuSiQueRecord(record_id="rec", question="Q", answer=f"answer{n_hops-1}",
                         answer_aliases=(), steps=steps, is_linear=True,
                         linear_failure="")


def final_state(events, mode):
    kept, _, _ = resolve_stream(events, mode)
    out = {}
    for e in kept:
        out[e.root_id] = e.payload_hash
    return out


# --- the task must not change ----------------------------------------------

@pytest.mark.parametrize("intervention", INTERVENTIONS)
@pytest.mark.parametrize("scope", SCOPES)
def test_gold_answer_and_hop_set_are_invariant(intervention, scope):
    r = record()
    base = base_stream(r)
    for m in MULTIPLICITIES:
        stream = build_stream(r, intervention, m, scope, seed=3)
        assert {e.hop for e in stream} == {e.hop for e in base}
        # The terminal target is a property of the record, never of the stream.
        assert r.steps[-1].answer == "answer3"


@pytest.mark.parametrize("intervention", INTERVENTIONS)
def test_lineage_distinct_root_count_is_constant(intervention):
    r = record()
    expected = len({e.root_id for e in base_stream(r)})
    for m in MULTIPLICITIES:
        for scope in SCOPES:
            stream = build_stream(r, intervention, m, scope, seed=5)
            assert stream_stats(stream)["n_roots"] == expected


def test_rechunking_and_rewording_never_mint_a_new_root():
    r = record()
    roots = {e.root_id for e in base_stream(r)}
    pool = {f"rec:p3": ["A completely different wording of the third paragraph."]}
    for intervention in ("overlap", "paraphrase"):
        stream = build_stream(r, intervention, 16, "all_hops", seed=1,
                              paraphrases=pool)
        assert {e.root_id for e in stream} == roots
        # ...and they are alternatives, not refinements.
        assert {e.version for e in stream} == {0}


def test_exact_copies_are_byte_identical():
    r = record()
    stream = build_stream(r, "exact", 16, "one_hop", seed=0)
    duplicated = [e for e in stream if e.hop == r.n_hops - 1]
    assert len(duplicated) == 16
    assert len({e.paragraph for e in duplicated}) == 1
    assert len({e.payload_hash for e in duplicated}) == 1
    assert len({version_key(e) for e in duplicated}) == 1


def test_overlapping_views_are_spans_of_the_source():
    words = record().steps[0].paragraph_text.split()
    views = overlapping_views(" ".join(words), 6)
    assert len(views) == 6
    assert len(set(views)) > 1, "views must actually differ"
    for view in views:
        assert set(view.split()) <= set(words), "a view invented text"


# --- re-chunking with the whole paragraph ------------------------------------

@pytest.mark.parametrize("words", (21, 40, 150))
def test_rechunk_views_are_distinct_spans_led_by_the_paragraph(words):
    text = record(words=words).steps[0].paragraph_text
    source = text.split()
    views = overlapping_views(text, 16)
    assert len(views) == 16 and len(set(views)) == 16
    assert views[0] == text, "the original delivery must come first"
    joined = " " + " ".join(source) + " "
    for view in views[1:]:
        assert len(view.split()) < len(source)
        assert " " + view + " " in joined, "a view is not a contiguous span"


def test_rechunk_views_at_lower_multiplicity_are_a_prefix():
    text = record(words=40).steps[0].paragraph_text
    full = overlapping_views(text, 16)
    for m in MULTIPLICITIES:
        assert overlapping_views(text, m) == full[:m]


def test_rechunk_ledger_holds_the_whole_paragraph_and_drops_every_window():
    r = record(n_hops=2, words=40)
    base = base_stream(r)
    for m in MULTIPLICITIES:
        stream = build_stream(r, "rechunk", m, "all_hops", seed=0)
        kept, credit, suppressed = resolve_stream(stream, "max")
        assert kept == base, "the ledger must reproduce the clean stream"
        assert sum(credit.values()) == len(base)
        assert suppressed == len(stream) - len(base)
        # Every window reaches a provenance-free reader and exact hashing.
        assert len(resolve_stream(stream, "none")[0]) == len(stream)
        assert sum(resolve_stream(stream, "canonical")[1].values()) == len(stream)


def test_rechunk_final_state_is_order_invariant():
    r = record(n_hops=2, words=40)
    stream = build_stream(r, "rechunk", 16, "all_hops", seed=0)
    states = set()
    for t in range(12):
        perm = stream[:]
        random.Random(t).shuffle(perm)
        states.add(tuple(sorted(final_state(perm, "max").items())))
    assert states == {tuple(sorted(final_state(base_stream(r), "max").items()))}


def test_fragments_withhold_the_paragraph_and_conserve_credit():
    r = record(n_hops=2, words=40)
    base = base_stream(r)
    for m in MULTIPLICITIES[1:]:
        stream = build_stream(r, "fragments", m, "all_hops", seed=0)
        for e in base:
            views = [x.paragraph for x in stream if x.root_id == e.root_id]
            assert len(views) == m and len(set(views)) == m
            assert e.paragraph not in views, "the paragraph must be withheld"
            assert views == overlapping_views(e.paragraph, m + 1)[1:]
        kept, credit, _ = resolve_stream(stream, "max")
        assert sum(credit.values()) == len(base)
        assert sum(resolve_stream(stream, "none")[1].values()) == len(stream)
        # The ledger's final state is one fragment per root, never the paragraph.
        state = final_state(stream, "max")
        assert set(state.values()).isdisjoint({e.payload_hash for e in base})


def test_fragments_final_state_is_order_invariant():
    r = record(n_hops=2, words=40)
    stream = build_stream(r, "fragments", 16, "all_hops", seed=0)
    states = set()
    for t in range(12):
        perm = stream[:]
        random.Random(t).shuffle(perm)
        states.add(tuple(sorted(final_state(perm, "max").items())))
    assert len(states) == 1


def test_frozen_overlap_construction_is_unchanged():
    """The frozen sweeps used at most three partial windows; keep it so."""
    text = record(words=40).steps[0].paragraph_text
    views = partial_windows(text, 16)
    assert len(set(views)) <= 3 and text not in views
    stream = build_stream(record(), "overlap", 16, "one_hop", seed=0)
    assert all(e.version == 0 for e in stream)


# --- accounting -------------------------------------------------------------

@pytest.mark.parametrize("intervention", INTERVENTIONS)
def test_ec_conserves_and_provenance_free_grows(intervention):
    r = record()
    ceiling = len({e.root_id for e in base_stream(r)})
    grew = False
    for m in MULTIPLICITIES:
        stream = build_stream(r, intervention, m, "all_hops", seed=7)
        ec = sum(resolve_stream(stream, "max")[1].values())
        first = sum(resolve_stream(stream, "first")[1].values())
        plain = sum(resolve_stream(stream, "none")[1].values())
        assert ec == ceiling, f"{intervention} x{m}: EC credited {ec}"
        assert first == ceiling
        assert plain >= ec
        if m > 1 and plain > ec:
            grew = True
    assert grew, f"{intervention}: provenance-free accounting never grew"


def test_exact_redelivery_applies_no_second_transformation():
    """Under EC the ledger drops a re-arrival, so the content path sees the
    chain exactly once however many copies arrive."""
    r = record()
    base = base_stream(r)
    for m in MULTIPLICITIES:
        stream = build_stream(r, "exact", m, "all_hops", seed=0)
        kept, credit, suppressed = resolve_stream(stream, "max")
        assert [e.payload_hash for e in kept] == [e.payload_hash for e in base]
        assert suppressed == len(stream) - len(base)
    # ...whereas provenance-free accounting reprocesses every copy.
    stream = build_stream(r, "exact", 16, "all_hops", seed=0)
    assert len(resolve_stream(stream, "none")[0]) == len(stream)


def test_only_the_max_join_is_order_invariant_under_rechunking():
    """The final ledger state must not depend on which chunk arrived first."""
    r = record()
    stream = build_stream(r, "overlap", 8, "one_hop", seed=1)
    distinct = {}
    for mode in MODES:
        states = set()
        for t in range(12):
            perm = stream[:]
            random.Random(t).shuffle(perm)
            states.add(tuple(sorted(final_state(perm, mode).items())))
        distinct[mode] = len(states)
    assert distinct["max"] == 1, "EC must be order-invariant"
    assert distinct["first"] > 1, "the first-version ablation is order-dependent"
    assert distinct["canonical"] > 1, "canonical dedup is order-dependent"


def test_canonical_dedup_collapses_exact_copies_but_not_chunks():
    r = record()
    exact = build_stream(r, "exact", 16, "one_hop", seed=0)
    overlap = build_stream(r, "overlap", 16, "one_hop", seed=0)
    ceiling = len({e.root_id for e in base_stream(r)})
    assert sum(resolve_stream(exact, "canonical")[1].values()) == ceiling
    assert sum(resolve_stream(overlap, "canonical")[1].values()) > ceiling


def test_ledger_mode_mapping():
    assert ledger_mode("full") == "max"
    assert ledger_mode("transformer_ec") == "max"
    assert ledger_mode("filter_only") == "first"
    assert ledger_mode("transformer_filter_only") == "first"
    assert ledger_mode("canonical_dedup") == "canonical"
    for v in ("plain", "no_provenance", "transformer_plain"):
        assert ledger_mode(v) == "none"
    assert ledger_mode("dempster_fusion") == "dempster"
    assert ledger_mode("covariance_intersection") == "covariance_intersection"
    # Every ledger mode is one of the original four, a near-duplicate detector
    # or a fusion rule; nothing may be added without a variant that reaches it.
    assert set(LEDGER_MODES) == set(MODES) | set(DETECTORS) | set(FUSION_MODES)
    assert set(DEDUP_VARIANTS.values()) == {"canonical"} | set(DETECTORS)
    assert set(FUSION_VARIANTS.values()) == set(FUSION_MODES)


# --- fusion rules: the two ways to get evidence accounting wrong -------------

def test_dempster_is_not_idempotent_and_saturates():
    """The defining defect. Dempster's rule sharpens belief when a source is
    combined with itself, so credit rises with repetition instead of holding,
    and it saturates at 1 / mass rather than growing without bound."""
    assert dempster_credit(1) == pytest.approx(1.0)
    seq = [dempster_credit(n) for n in (1, 2, 4, 8, 16)]
    assert all(b > a for a, b in zip(seq, seq[1:])), seq
    assert seq[-1] < 1.0 / DEMPSTER_MASS + 1e-9
    assert dempster_credit(10_000) == pytest.approx(1.0 / DEMPSTER_MASS)


def test_fusion_rules_suppress_nothing():
    """Both admit the whole stream, so any difference from `plain` is the
    accounting rule and not a different content path."""
    r = record()
    events = build_stream(r, "exact", 16, "one_hop", seed=0)
    for mode in FUSION_MODES:
        kept, _, suppressed = resolve_stream(events, mode)
        assert suppressed == 0
        assert len(kept) == len(events)


@pytest.mark.parametrize("intervention", ("exact", "overlap", "cycle"))
def test_dempster_inflates_and_covariance_intersection_undercredits(intervention):
    """Neither fusion rule gets both axes right, and they fail oppositely.

    The ledger is the only arm that both holds credit at the ceiling under
    repetition and still credits every lineage-distinct root."""
    r = record()
    events = build_stream(r, intervention, 16, "one_hop", seed=0)
    ceiling = len({e.root_id for e in base_stream(r)})

    ec = sum(resolve_stream(events, "max")[1].values())
    none = sum(resolve_stream(events, "none")[1].values())
    dst = sum(resolve_stream(events, "dempster")[1].values())
    ci_credit = resolve_stream(events, "covariance_intersection")[1]
    ci = sum(ci_credit.values())

    assert ec == pytest.approx(ceiling)
    # Dempster inflates above the ceiling, but saturates below no defence.
    assert dst > ceiling
    assert dst < none
    # Covariance intersection never lets one root exceed its due share, which
    # is what makes it duplicate-safe, and is also why it under-credits: the
    # convex weights are split across lineage-distinct roots.
    assert max(ci_credit.values()) <= 1.0 + 1e-9
    assert ci < ceiling


def test_covariance_intersection_is_duplicate_safe_on_a_single_root():
    """With one root there is nothing to divide, so the convex constraint gives
    exactly the right answer however many times it arrives."""
    r = record(n_hops=1)
    for mult in (1, 2, 4, 16):
        events = build_stream(r, "exact", mult, "one_hop", seed=0)
        credit = resolve_stream(events, "covariance_intersection")[1]
        assert sum(credit.values()) == pytest.approx(1.0)


# --- no label leakage through the intervention ------------------------------

def test_multiplicity_and_length_carry_no_label_information():
    """Stream length, padding and delivery position are functions of the
    intervention alone, so they cannot identify the answer."""
    answers = ["answer3", "totally other"]
    lengths = set()
    for answer in answers:
        r = record()
        steps = list(r.steps)
        steps[-1] = MuSiQueStep(question=steps[-1].question, answer=answer,
                                paragraph_title=steps[-1].paragraph_title,
                                paragraph_text=steps[-1].paragraph_text,
                                paragraph_idx=steps[-1].paragraph_idx,
                                dependencies=steps[-1].dependencies)
        alt = MuSiQueRecord(record_id=r.record_id, question=r.question,
                            answer=answer, answer_aliases=(), steps=tuple(steps),
                            is_linear=True, linear_failure="")
        for intervention in INTERVENTIONS:
            for m in MULTIPLICITIES:
                stream = build_stream(alt, intervention, m, "one_hop", seed=0)
                lengths.add((intervention, m, len(stream)))
    by_condition = {}
    for intervention, m, n in lengths:
        by_condition.setdefault((intervention, m), set()).add(n)
    for key, values in by_condition.items():
        assert len(values) == 1, f"{key} length varies with the label: {values}"


def test_interventions_are_deterministic_in_their_seed():
    r = record()
    for intervention in INTERVENTIONS:
        a = build_stream(r, intervention, 8, "one_hop", seed=11)
        b = build_stream(r, intervention, 8, "one_hop", seed=11)
        assert [(e.hop, e.payload_hash, e.version) for e in a] == \
               [(e.hop, e.payload_hash, e.version) for e in b]


# --- calibration ------------------------------------------------------------

def test_calibration_evidence_coefficient_is_non_negative():
    torch.manual_seed(0)
    sim = torch.randn(200, 20)
    target = torch.randint(0, 20, (200,))
    credit = torch.rand(200) * 4
    cal = fit(sim, target, credit, "evidence", iters=120)
    assert cal.b >= 0.0, "evidence may only sharpen, never blunt"


def test_content_readout_ignores_credited_evidence():
    """Otherwise conservation would look successful tautologically."""
    torch.manual_seed(0)
    sim = torch.randn(50, 12)
    target = torch.randint(0, 12, (50,))
    cal = Calibration(log_temperature=0.1, a=1.0, raw_b=1.0)
    low = score(sim, target, torch.ones(50), cal, "content")
    high = score(sim, target, torch.full((50,), 9.0), cal, "content")
    assert low == high
    ev_low = score(sim, target, torch.ones(50), cal, "evidence")
    ev_high = score(sim, target, torch.full((50,), 9.0), cal, "evidence")
    assert ev_low != ev_high, "the evidence readout must depend on credit"


def test_ece_binning_is_the_preregistered_rule():
    conf = np.array([0.05, 0.35, 0.65, 0.95])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(conf, correct) == pytest.approx(
        np.mean([0.05, 0.35, 0.35, 0.05]), abs=1e-9)
    assert expected_calibration_error(np.array([]), np.array([])) != \
        expected_calibration_error(np.array([]), np.array([]))  # nan


def test_score_reports_every_required_metric():
    torch.manual_seed(0)
    m = score(torch.randn(30, 8), torch.randint(0, 8, (30,)), torch.ones(30),
              Calibration(), "content")
    for key in ("nll", "brier", "ece", "accuracy", "mean_max_confidence",
                "predictive_entropy", "mrr", "n", "n_candidates"):
        assert key in m


# --- integration guards on the evaluation path ------------------------------

def _tiny_cache(tmp_path):
    from ecnca.real.encode import EmbeddingCache, HashEncoder
    return EmbeddingCache(str(tmp_path / "c.npz"), HashEncoder(32))


def test_evaluation_does_not_mutate_cached_inputs(tmp_path):
    """A condition must not write back into the embedding cache it read."""
    from ecnca.real.musique_duplication import make_delivery_batch
    cache = _tiny_cache(tmp_path)
    r = record()
    stream = build_stream(r, "exact", 8, "one_hop", seed=0)
    kept, _, _ = resolve_stream(stream, "max")
    first = make_delivery_batch([kept], [r], cache)
    snapshot = {k: v.clone() for k, v in first.items() if torch.is_tensor(v)}
    second = make_delivery_batch([kept], [r], cache)
    for key, value in snapshot.items():
        assert torch.equal(value, second[key]), f"{key} changed between reads"
    # An exact copy must reuse the identical vector, not re-encode it.
    para = cache.encode([stream[0].paragraph])
    assert np.array_equal(para, cache.encode([stream[0].paragraph]))


def test_matched_variants_receive_identical_content_inputs(tmp_path):
    """Variants sharing a ledger mode must see byte-identical tensors."""
    from ecnca.real.musique_duplication import make_delivery_batch
    cache = _tiny_cache(tmp_path)
    r = record()
    stream = build_stream(r, "exact", 8, "all_hops", seed=0)
    batches = {}
    for variant in ("full", "transformer_ec"):
        kept, _, _ = resolve_stream(stream, ledger_mode(variant))
        batches[variant] = make_delivery_batch([kept], [r], cache)
    a, b = batches["full"], batches["transformer_ec"]
    for key in ("question", "paragraph", "answer", "mask", "hop_index"):
        assert torch.equal(a[key], b[key]), f"{key} differs between EC variants"


def test_duplicated_arrivals_reuse_their_hop_position(tmp_path):
    """A stream longer than the chain must not need new positional slots."""
    from ecnca.real.musique_duplication import make_delivery_batch
    from ecnca.real.transformer_transport import TransformerTransport
    cache = _tiny_cache(tmp_path)
    r = record()
    stream = build_stream(r, "exact", 16, "all_hops", seed=0)
    batch = make_delivery_batch([stream], [r], cache)
    assert batch["question"].shape[1] > 4, "stream should exceed the hop count"
    assert int(batch["hop_index"].max()) < 4, "hop indices must stay in range"
    model = TransformerTransport(emb_dim=32, hidden=16,
                                 variant="transformer_ec").eval()
    with torch.no_grad():
        out = model(batch)
    assert out["prediction"].shape == (1, 32)


# --- report-level guards ----------------------------------------------------

def test_report_is_complete_and_consistent():
    """The merged report must cover every cell with one candidate pool."""
    import json, pathlib
    path = pathlib.Path("results/musique_dup/report.json")
    if not path.is_file():
        pytest.skip("duplicated-hop sweep not present")
    d = json.loads(path.read_text())
    assert len(d["seeds"]) == 5
    assert len(d["variants"]) == 7
    seen = {(r["variant"], r["intervention"], r["scope"], r["multiplicity"])
            for r in d["rows"]}
    assert len(seen) == len(d["rows"]), "duplicate summary rows"
    for row in d["rows"]:
        assert row["n_seeds"] == 5, row


def test_conserving_variants_are_exactly_invariant_under_exact_redelivery():
    """The headline claim, asserted against the committed report."""
    import json, pathlib
    path = pathlib.Path("results/musique_dup/report.json")
    if not path.is_file():
        pytest.skip("duplicated-hop sweep not present")
    rows = {(r["variant"], r["intervention"], r["scope"], r["multiplicity"]): r
            for r in json.loads(path.read_text())["rows"]}
    for variant in ("full", "transformer_ec", "canonical_dedup", "filter_only"):
        base = rows[(variant, "exact", "one_hop", 1)]
        for m in (2, 4, 8, 16):
            cell = rows[(variant, "exact", "one_hop", m)]
            assert cell["prediction_drift"] == 0.0, (variant, m)
            assert cell["credited_over_ceiling"] == pytest.approx(1.0)
            for metric in ("content_accuracy", "evidence_nll", "evidence_ece",
                           "evidence_mean_max_confidence"):
                assert cell[metric] == pytest.approx(base[metric], abs=1e-9), \
                    f"{variant} {metric} moved at x{m}"
    # ...and provenance-free accounting does not survive the same test.
    for variant in ("plain", "transformer_plain"):
        cell = rows[(variant, "exact", "one_hop", 16)]
        assert cell["credited_over_ceiling"] > 1.5
        assert cell["prediction_drift"] > 0.0


def test_canonical_dedup_fails_on_overlapping_chunks_but_ec_does_not():
    import json, pathlib
    path = pathlib.Path("results/musique_dup/report.json")
    if not path.is_file():
        pytest.skip("duplicated-hop sweep not present")
    rows = {(r["variant"], r["intervention"], r["scope"], r["multiplicity"]): r
            for r in json.loads(path.read_text())["rows"]}
    ec = rows[("full", "overlap", "one_hop", 16)]["credited_over_ceiling"]
    canon = rows[("canonical_dedup", "overlap", "one_hop", 16)]["credited_over_ceiling"]
    assert ec == pytest.approx(1.0), ec
    assert canon > 1.0, canon


def test_paraphrase_rows_are_flagged_when_no_pool_was_generated():
    """Otherwise a degenerate condition could be read as a paraphrase result.

    A second run now supplies an audited pool and does license a paraphrase
    claim, so this no longer asserts that the paper makes no such claim
    anywhere. What it still must assert is that the rows of the run WITHOUT a
    pool are disclaimed where they are reported, and that the reader is sent to
    the run that does have one.
    """
    import json, pathlib
    path = pathlib.Path("results/musique_dup/report.json")
    if not path.is_file():
        pytest.skip("duplicated-hop sweep not present")
    d = json.loads(path.read_text())
    assert "paraphrase_pool_available" in d
    if not d["paraphrase_pool_available"]:
        assert d["paraphrase_records_covered"] == 0
        # The source is line-wrapped, so compare on collapsed whitespace.
        src = pathlib.Path("paper/main_v2.tex")
        if not src.is_file():
            src = pathlib.Path("paper/main.tex")
        text = " ".join(src.read_text().split()).lower()
        if not any(t in text for t in ("generated/tab_duplication",
                                       "generated/tab_hop_groups",
                                       "musique_dup/")):
            # The manuscript no longer reports this run, so there is nothing to
            # disclaim. It must then make no paraphrase claim about the
            # learned memory at all, in the setup or in the appendix.
            for start in ("subsection{duplication regimes for the learned memory}",
                          "section{duplicated-hop construction}"):
                i = text.find(start)
                if i >= 0:
                    section = text[i:text.find("section{", i + len(start))]
                    assert "paraphrase" not in section, start
            return
        assert "not a paraphrase result" in text, (
            "the pool-less run's paraphrase rows must be disclaimed where the "
            "manuscript counts them")
        assert "app:sybil" in text or "app:iclr" in text, (
            "the disclaimer must point at the audited paraphrase attack, or a "
            "reader cannot tell which paraphrase numbers are real")
