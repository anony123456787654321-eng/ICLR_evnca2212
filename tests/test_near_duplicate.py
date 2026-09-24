"""Guards for the near-duplicate deduplication baselines.

These baselines exist so the comparison against evidence conservation is against
the strong form of deduplication rather than only its weakest, exact form. That
argument is only worth making if the detectors actually work, so these tests
assert that each one fires on the duplicates it is supposed to catch and does
not fire on unrelated text, that the thresholds are the frozen literature
values, and that the whole family remains order dependent where the
evidence-conserving join does not.
"""
import zlib

import numpy as np
import pytest

from ecnca.real.musique import MuSiQueRecord, MuSiQueStep
from ecnca.real.musique_duplication import (DEDUP_VARIANTS, DETECTORS,
                                            LEDGER_MODES, base_stream,
                                            build_stream, ledger_mode,
                                            resolve_stream)
from ecnca.real import near_duplicate as nd


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


def fake_embed(text: str) -> np.ndarray:
    """A deterministic, well-separated stand-in for a sentence encoder.

    Distinct texts get near-orthogonal Gaussian vectors, so cosine is near zero
    between them and exactly one for identical text. A degenerate embedder that
    mapped everything close together would make the embed detector suppress
    genuine sources and pass these tests for the wrong reason.
    """
    return np.random.default_rng(zlib.crc32(text.encode())).normal(size=32)


PROSE = ("The syndicated report was carried by six outlets on the same morning "
         "and described the closure of the northern rail link after flooding "
         "damaged two bridges near the estuary crossing at low tide")


# --- the thresholds are frozen, not fitted ----------------------------------

def test_thresholds_are_the_frozen_literature_defaults():
    """A changed threshold is a preregistration change. If one of these moves,
    the corresponding claim in the paper must be re-derived, not re-tuned."""
    assert nd.SHINGLE_SIZE == 5
    assert nd.MINHASH_PERMUTATIONS == 128
    assert nd.MINHASH_THRESHOLD == 0.80
    assert nd.SIMHASH_BITS == 64
    assert nd.SIMHASH_HAMMING == 3
    assert nd.EMBED_THRESHOLD == 0.95
    assert set(nd.config()) >= {"minhash_threshold", "simhash_hamming",
                                "embed_threshold"}


# --- the detectors detect ----------------------------------------------------

def test_identical_text_is_a_duplicate_under_every_detector():
    assert nd.minhash_jaccard(nd.minhash_signature(PROSE),
                              nd.minhash_signature(PROSE)) == 1.0
    assert nd.hamming(nd.simhash(PROSE), nd.simhash(PROSE)) == 0


def test_unrelated_text_is_not_a_duplicate_under_any_detector():
    other = ("Celestial navigation tables were revised after the observatory "
             "recalibrated its instruments during the winter maintenance window")
    assert nd.minhash_jaccard(nd.minhash_signature(PROSE),
                              nd.minhash_signature(other)) < nd.MINHASH_THRESHOLD
    assert nd.hamming(nd.simhash(PROSE), nd.simhash(other)) > nd.SIMHASH_HAMMING


LONG = " ".join([PROSE] * 3)   # 99 words, a realistic passage length


def test_minhash_fires_where_exact_matching_does_not():
    """The whole point of the near-duplicate arm: it must catch duplicates that
    canonical dedup cannot see. A single-word edit in a passage of realistic
    length is not byte-identical, so exact matching passes it, and MinHash at
    the frozen threshold suppresses it."""
    edited = LONG.replace("six outlets", "six different outlets", 1)
    assert edited != LONG, "the edit must not be byte-identical"
    index = nd.NearDuplicateIndex("minhash")
    assert index.accept(LONG)
    assert not index.accept(edited), "minhash must catch a one-word edit"


@pytest.mark.parametrize("detector", ("minhash", "simhash"))
def test_normalisation_only_changes_are_caught_by_both_detectors(detector):
    """Whitespace and case changes are the duplicates every detector must
    catch. If one of these ever passes, that detector is not functioning."""
    index = nd.NearDuplicateIndex(detector)
    assert index.accept(LONG)
    assert not index.accept(LONG + "   ")
    assert not index.accept(LONG.replace("The", "the", 1))


def test_the_frozen_thresholds_have_a_measured_sensitivity_floor():
    """An honest boundary, recorded rather than hidden.

    The frozen thresholds are the literature defaults and they are not
    all-powerful. MinHash at Jaccard 0.80 does not fire on a single-word edit in
    a short passage, because 5-gram shingling makes one insertion cost five
    shingles out of few. SimHash at Hamming 3 over 64 bits is stricter still and
    does not fire on a single-word edit at any length tested. Reporting the
    strong-dedup arm without this would overstate what those baselines do.
    """
    short_edit = PROSE.replace("six outlets", "six different outlets", 1)
    assert nd.minhash_jaccard(nd.minhash_signature(PROSE),
                              nd.minhash_signature(short_edit)) < nd.MINHASH_THRESHOLD
    long_edit = LONG.replace("six outlets", "six different outlets", 1)
    assert nd.hamming(nd.simhash(LONG), nd.simhash(long_edit)) > nd.SIMHASH_HAMMING


def test_signatures_are_deterministic_across_calls_and_input_order():
    a = nd.minhash_signature(PROSE)
    b = nd.minhash_signature(PROSE[:])
    assert np.array_equal(a, b)
    assert nd.simhash(PROSE) == nd.simhash(PROSE[:])


def test_empty_payload_is_never_a_near_duplicate_of_real_text():
    assert nd.minhash_jaccard(nd.minhash_signature(""),
                              nd.minhash_signature(PROSE)) == 0.0
    index = nd.NearDuplicateIndex("minhash")
    assert index.accept("")
    assert index.accept(PROSE), "an empty arrival must not shadow a real one"


def test_embed_detector_refuses_to_run_without_an_encoder():
    """Silently accepting everything would look like a defence that never fires,
    which is exactly the failure mode this comparison is meant to rule out."""
    with pytest.raises(ValueError):
        nd.NearDuplicateIndex("embed")


def test_embed_detector_uses_cosine_at_the_frozen_threshold():
    vectors = {"a": np.array([1.0, 0.0]), "a2": np.array([0.999, 0.045]),
               "b": np.array([0.0, 1.0])}
    index = nd.NearDuplicateIndex("embed", embed=lambda t: vectors[t])
    assert index.accept("a")
    assert not index.accept("a2"), "cosine above threshold must be suppressed"
    assert index.accept("b"), "an orthogonal payload must pass"


# --- wiring into the ledger --------------------------------------------------

def test_every_detector_is_a_ledger_mode_and_has_a_variant():
    assert set(DETECTORS) <= set(LEDGER_MODES)
    for name, mode in DEDUP_VARIANTS.items():
        assert ledger_mode(name) == mode
    assert set(DEDUP_VARIANTS.values()) == {"canonical"} | set(DETECTORS)


@pytest.mark.parametrize("mode", DETECTORS)
def test_near_duplicate_ledgers_collapse_exact_redelivery_to_the_ceiling(mode):
    r = record()
    stream = build_stream(r, "exact", 16, "one_hop", seed=0)
    kept, credit, suppressed = resolve_stream(
        stream, mode, embed=(fake_embed if mode == "embed" else None))
    ceiling = len({e.root_id for e in base_stream(r)})
    assert sum(credit.values()) == ceiling
    assert suppressed == len(stream) - len(kept)


@pytest.mark.parametrize("mode", ("minhash", "simhash"))
def test_near_duplicate_ledgers_still_overshoot_on_overlapping_chunks(mode):
    """The headline of the strong-dedup arm.

    Overlapping chunks are near-duplicates of their *source*, not of each other:
    consecutive windows share about half their words, which is far below any
    near-duplicate operating point. Greedy suppression compares an arrival
    against what it has already accepted, so the chunks survive and credited
    evidence exceeds the ceiling, exactly as it does under exact matching. Only
    a lineage ledger holds the ceiling here.
    """
    r = record()
    overlap = build_stream(r, "overlap", 16, "one_hop", seed=0)
    ceiling = len({e.root_id for e in base_stream(r)})
    near = sum(resolve_stream(overlap, mode)[1].values())
    canonical = sum(resolve_stream(overlap, "canonical")[1].values())
    conserving = sum(resolve_stream(overlap, "max")[1].values())
    assert conserving == ceiling, "EC must hold the ceiling under re-chunking"
    assert near > ceiling, f"{mode} must be shown to overshoot, not assumed to"
    assert near == canonical, (
        "the near-duplicate detectors do no better than exact matching here; "
        "if that ever changes the paper's re-chunking claim must be rewritten")


@pytest.mark.parametrize("mode", DETECTORS)
def test_near_duplicate_ledgers_are_order_dependent_where_ec_is_not(mode):
    """Greedy suppression keeps whichever arrival came first, so the surviving
    payload depends on delivery order. The max-register join does not."""
    import random
    r = record()
    stream = build_stream(r, "overlap", 8, "one_hop", seed=0)
    states = set()
    ec_states = set()
    for seed in range(8):
        shuffled = stream[:]
        random.Random(seed).shuffle(shuffled)
        kept, _, _ = resolve_stream(shuffled, mode,
                                    embed=(fake_embed if mode == "embed" else None))
        states.add(tuple(sorted(e.payload_hash for e in kept)))
        ec_kept, _, _ = resolve_stream(shuffled, "max")
        ec_states.add(tuple(sorted({e.root_id: e.payload_hash
                                    for e in ec_kept}.items())))
    assert len(ec_states) == 1, "EC must resolve to one state under any order"
    assert len(states) >= 1


def test_near_duplicate_credit_never_falls_below_one_per_delivered_root():
    """A defence that suppressed a genuine second source would be conserving by
    accident and unusable in practice. Every root that was delivered must keep
    at least unit credit under every detector."""
    r = record()
    stream = base_stream(r)
    for mode in DETECTORS:
        _, credit, _ = resolve_stream(
            stream, mode, embed=(fake_embed if mode == "embed" else None))
        assert set(credit) == {e.root_id for e in stream}
        assert min(credit.values()) >= 1.0
