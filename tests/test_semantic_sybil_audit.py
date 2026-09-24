import numpy as np

from analysis.semantic_sybil_audit import (choose_semantic_threshold,
                                            make_attack_records,
                                            retain_candidates)


class TinyCache:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if text in ("source report here", "independent wording here"):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


def test_semantic_threshold_respects_false_link_budget():
    threshold, fit = choose_semantic_threshold(
        [("source report here", "independent wording here")],
        [("source report here", "unrelated evidence")], TinyCache())
    assert threshold == 1.0
    assert fit["train_false_link_rate"] == 0.0


def test_retention_and_semantic_defense_share_one_root():
    row = {"record_id": "r", "claim": "c", "label": "Supported",
           "source_text": "source report here",
           "candidates": [{"style_id": i, "text": "independent wording here " + str(i)}
                          for i in range(4)]}
    cache = TinyCache()
    # Make all generated candidates semantically identical to the source while
    # keeping their surface forms distinct and below the near-copy threshold.
    cache.encode = lambda texts: np.tile(np.array([[1.0, 0.0]], dtype=np.float32),
                                         (len(texts), 1))
    reports = retain_candidates([row], cache)
    assert reports[0]["eligible"]
    url = make_attack_records(reports, 4, "url", 0.9)[0]
    semantic = make_attack_records(reports, 4, "semantic", 0.9)[0]
    assert len({p.root_id for p in url.passages}) == 4
    assert len({p.root_id for p in semantic.passages}) == 1


def test_non_entailing_candidate_is_rejected_even_when_embedding_matches():
    row = {"record_id": "r", "claim": "c", "label": "Supported",
           "source_text": "source report here",
           "candidates": [{"style_id": 0, "text": "independent wording here"}]}
    cache = TinyCache()
    nli = {(0, 0, "source_to_candidate"): {"label": "entailment", "probabilities": {}},
           (0, 0, "candidate_to_source"): {"label": "neutral", "probabilities": {}}}
    report = retain_candidates([row], cache, nli=nli)[0]
    assert not report["candidates"][0]["retained"]
    assert "not_bidirectional_entailment" in report["candidates"][0]["reasons"]
