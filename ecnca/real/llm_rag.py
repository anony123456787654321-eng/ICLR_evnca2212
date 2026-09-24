"""Lineage-bounded confidence for an LLM retrieval-augmented verdict.

This is the deployment-shaped test of the paper's claim. A frozen open-weights
instruct model reads a claim and its retrieved passages and produces a
distribution over the four AVeriTeC verdicts. Nothing about the language model
is trained, and no proprietary API is used.

What the ledger changes is not the language model but the confidence attached to
its answer. Under Assumption A6 the readout scales natural parameters by a
monotone function of credited evidence, and for a categorical head that is
exactly a scaling of the logits:

    logits' = logits * (E_credited / T)

with `T` one scalar per defence, fitted once on a calibration split and frozen.
The defences differ only in how `E_credited` is counted:

    none        every retrieved passage counts                (the RAG default)
    canonical   distinct passage texts count
    minhash     passages surviving MinHash near-duplicate suppression
    simhash     passages surviving SimHash suppression
    embed       passages surviving embedding-cosine suppression
    ec          distinct lineage roots count                  (this paper)

Because each defence gets its own fitted `T`, none is advantaged by the mean
level of its count: every difference that survives comes from how the count
varies *across claims*, which is precisely the quantity lineage is supposed to
get right.

Two evaluation modes are supported and reported separately.

`accounting` gives every defence a byte-identical prompt, so the model's answer
is identical and accuracy at full coverage is identical by construction. Any
difference in accuracy at partial coverage is therefore attributable to the
confidence ranking alone. This is the clean causal test.

`pipeline` lets each defence filter the passages that reach the prompt, which is
what a deployed system would do. Predictions then differ too, so the comparison
is realistic but no longer isolates the confidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np

from .near_duplicate import NearDuplicateIndex

LABELS = ["Supported", "Refuted", "Not Enough Evidence",
          "Conflicting Evidence/Cherrypicking"]
# `ec` keeps every passage and bounds only the credit, which is the method.
# `ec_filter` keeps one passage per source, which is source filtering; it is
# retained as a labelled ablation because the paper's own argument is that
# filtering is the wrong design, and an earlier version of this file
# implemented the filter under the name `ec`.
DEFENCES = ("none", "canonical", "minhash", "simhash", "embed", "ec",
            "ec_filter")
CREDIT_ONLY = ("ec",)

# Single-token choice letters mapped to the four verdicts. Scoring one letter
# avoids the multi-token label problem entirely: a first-token score over
# "Supported"/"Refuted"/"Not Enough Evidence"/"Conflicting..." is not a verdict
# likelihood, and the labels do not tokenise to a clean one-token prefix set.
CHOICES = ("A", "B", "C", "D")
INSTRUCTION = (
    "You are a fact-checking assistant. Using only the evidence provided, "
    "decide the verdict for the claim.\n"
    "A. Supported\n"
    "B. Refuted\n"
    "C. Not Enough Evidence\n"
    "D. Conflicting Evidence/Cherrypicking\n"
    "Reply with the single letter A, B, C or D and nothing else.")
BLOCK = "\nClaim: {claim}\n\nEvidence:\n{evidence}\n\nAnswer:{verdict}"


def format_block(claim: str, evidence: str, verdict: str = "") -> str:
    """One claim block. `verdict` is a choice LETTER for a demonstration."""
    if verdict and verdict in LABELS:
        verdict = CHOICES[LABELS.index(verdict)]
    return BLOCK.format(claim=claim.strip(), evidence=evidence,
                        verdict=(" " + verdict) if verdict else "")


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved passage with the lineage the store already supplies.

    AVeriTeC stores gold evidence as question-answer pairs and retrieved
    evidence as bare sentences. `display` renders both the same way so a
    demonstration drawn from the gold side does not arrive in a different format
    from the query it is meant to demonstrate.
    """
    text: str
    root_id: str
    content_hash: str
    question: str = ""

    @property
    def display(self) -> str:
        q, t = self.question.strip(), self.text.strip()
        if not q:
            return t
        if not q.endswith("?"):
            q += "?"
        return f"{q} {t}"


def credited_evidence(items: Sequence[EvidenceItem], defence: str,
                      embed: Callable[[str], np.ndarray] | None = None) -> float:
    """How much evidence a defence is willing to credit.

    `ec` reads lineage and counts each source document once however many
    passages carry it. The near-duplicate defences read payloads only, so two
    chunks of one document are two units of evidence to them unless the chunks
    happen to look alike.
    """
    if not items:
        return 0.0
    if defence == "none":
        return float(len(items))
    if defence in ("ec", "ec_filter"):
        return float(len({i.root_id for i in items}))
    if defence == "canonical":
        return float(len({i.content_hash for i in items}))
    if defence in ("minhash", "simhash", "embed"):
        index = NearDuplicateIndex(defence, embed=embed)
        return float(sum(1 for i in items if index.accept(i.text)))
    raise ValueError(f"unknown defence {defence!r}")


def surviving_items(items: Sequence[EvidenceItem], defence: str,
                    embed: Callable[[str], np.ndarray] | None = None
                    ) -> List[EvidenceItem]:
    """Which passages a defence would place in the prompt (`pipeline` mode).

    Evidence conservation places EVERY passage in the prompt. The whole claim of
    the method is that computation and content may circulate freely while only
    credited evidence is bounded, so a variant that drops complementary
    paragraphs from one source is not this method: it is the first-version
    filter the paper argues against, and it is available as `ec_filter`.
    """
    if defence in ("none",) + CREDIT_ONLY:
        return list(items)
    if defence == "ec_filter":
        # One passage per lineage root: the ledger's max-register join keeps a
        # single version of each source, deterministically by content hash so
        # the choice does not depend on retrieval order.
        # Selection is by VALUE, not object identity: the same EvidenceItem
        # object delivered four times previously survived four times, because
        # `best[root] is item` held for each occurrence.
        best: Dict[str, str] = {}
        for i in items:
            cur = best.get(i.root_id)
            if cur is None or i.content_hash > cur:
                best[i.root_id] = i.content_hash
        out, taken = [], set()
        for i in items:
            if i.content_hash == best[i.root_id] and i.root_id not in taken:
                taken.add(i.root_id)
                out.append(i)
        return out
    if defence == "canonical":
        seen, out = set(), []
        for i in items:
            if i.content_hash in seen:
                continue
            seen.add(i.content_hash)
            out.append(i)
        return out
    index = NearDuplicateIndex(defence, embed=embed)
    return [i for i in items if index.accept(i.text)]


def fit_evidence(items: Sequence[EvidenceItem], max_chars: int
                 ) -> List[EvidenceItem]:
    """The prefix of `items` that fits the character budget.

    Credit must be computed from this, not from the full retrieved list. A
    previous version credited two sources for a prompt whose evidence block read
    "(none)", because truncation happened after counting. Returning the surviving
    items lets the caller count what the model actually sees.
    """
    out, used = [], 0
    for n, item in enumerate(items, 1):
        piece = f"[{n}] {item.display}"
        if used + len(piece) > max_chars:
            break
        out.append(item)
        used += len(piece)
    return out


def render_evidence(items: Sequence[EvidenceItem], max_chars: int) -> str:
    kept = fit_evidence(items, max_chars)
    lines = [f"[{n}] {i.display}" for n, i in enumerate(kept, 1)]
    return "\n".join(lines) if lines else "(none)"


def build_prompt(claim: str, items: Sequence[EvidenceItem],
                 max_chars: int = 6000,
                 demonstrations: Sequence[str] = ()) -> str:
    """Instruction, optional frozen demonstrations, then the query block.

    Demonstrations are drawn once from the training split and are byte-identical
    across defences and across records, so they cannot advantage any defence and
    cannot leak evaluation content. Zero-shot verdicts on this task fall below
    the majority-class baseline, which makes a model's confidence useless to
    rank; a few demonstrations are what buy the signal the experiment needs.
    """
    parts = [INSTRUCTION]
    parts.extend(demonstrations)
    parts.append(format_block(claim, render_evidence(items, max_chars)))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# calibration and scoring
# ---------------------------------------------------------------------------

def scaled_probs(logits: np.ndarray, evidence: np.ndarray,
                 temperature: float) -> np.ndarray:
    """Apply the evidence-scaled readout of Assumption A6.

    `logits` is (N, L) and `evidence` is (N,). The multiplier is
    `evidence / temperature`. Zero credited evidence yields an exactly uniform
    distribution, which is the only defensible readout when nothing is credited.

    A previous version floored the multiplier at 1e-3 and called the result
    uniform. It is not: with large logits that floor left 99.96% confidence at
    zero credit. The multiplier is now set to exactly zero there, and the guard
    is `test_zero_credit_is_exactly_uniform`.
    """
    lam = np.asarray(evidence, dtype=float) / float(temperature)
    lam = np.where(lam > 0.0, lam, 0.0)
    z = np.asarray(logits, dtype=float) * lam[:, None]
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, evidence: np.ndarray, labels: np.ndarray,
                    grid: Sequence[float] | None = None,
                    return_diagnostics: bool = False):
    """One scalar per defence, chosen on the calibration split by NLL.

    Fitting per defence is deliberate: it removes any advantage a defence could
    get from the mean level of its count, so what remains is only how that count
    varies from claim to claim.
    """
    if grid is None:
        grid = np.exp(np.linspace(math.log(0.01), math.log(5000.0), 400))
    grid = np.asarray(grid, dtype=float)
    keep = labels >= 0
    if keep.sum() == 0:
        return (1.0, {"at_boundary": True, "nll": float("nan")}) if return_diagnostics else 1.0
    best, best_nll = float(grid[0]), float("inf")
    for t in grid:
        p = scaled_probs(logits[keep], evidence[keep], float(t))
        nll = -np.log(np.clip(p[np.arange(keep.sum()), labels[keep]], 1e-12, None)).mean()
        if nll < best_nll:
            best, best_nll = float(t), float(nll)
    # A fit at either end of the grid means the optimum was not bracketed and
    # the comparison between defences would be reading the grid, not the data.
    at_boundary = bool(np.isclose(best, grid[0]) or np.isclose(best, grid[-1]))
    if return_diagnostics:
        return best, {"at_boundary": at_boundary, "nll": best_nll}
    return best


# ---------------------------------------------------------------------------
# selective prediction
# ---------------------------------------------------------------------------

def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Accuracy at every coverage, plus the area under the risk--coverage curve.

    Examples are answered in decreasing confidence order. Where confidences tie,
    the answered set at a coverage inside the tie group is not determined, so we
    report the EXPECTED accuracy over orderings of that group rather than one
    arbitrary ordering. Concretely, every member of a tie group is assigned the
    group's mean correctness before the cumulative sum, which is the expectation
    over uniformly random tie-breaks and makes the curve invariant to the order
    the records arrived in.

    A previous implementation broke ties with a fixed permutation of positions,
    which is not order-invariant: reversing the input changed AURC from 0.6244
    to 0.3756 on eight tied examples. The regression guard is
    `test_ties_are_permutation_invariant`.

    Lower AURC is better.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool).astype(float)
    n = confidence.size
    if n == 0:
        return {"aurc": float("nan"), "coverage": [], "accuracy": []}
    order = np.argsort(-confidence, kind="stable")
    conf_sorted = confidence[order]
    hit = correct[order]
    # Replace each tie group by its mean correctness: the expectation over
    # random orderings within the group.
    start = 0
    for i in range(1, n + 1):
        if i == n or conf_sorted[i] != conf_sorted[start]:
            if i - start > 1:
                hit[start:i] = hit[start:i].mean()
            start = i
    cum = np.cumsum(hit)
    k = np.arange(1, n + 1)
    acc = cum / k
    return {"aurc": float(np.mean(1.0 - acc)),
            "coverage": (k / n).tolist(), "accuracy": acc.tolist()}


def accuracy_at_coverage(confidence: np.ndarray, correct: np.ndarray,
                         coverages: Sequence[float]) -> Dict[str, float]:
    rc = risk_coverage(confidence, correct)
    acc = np.asarray(rc["accuracy"])
    n = acc.size
    out = {}
    for c in coverages:
        idx = max(0, min(n - 1, int(round(c * n)) - 1))
        out[f"acc@{c:g}"] = float(acc[idx])
    out["aurc"] = rc["aurc"]
    return out


def auroc_with_ci(confidence: np.ndarray, correct: np.ndarray,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC with a bootstrap interval, and undefined handled explicitly.

    A point AUROC on a few dozen examples carries an interval wide enough to
    span chance either way, so reporting it bare invites exactly the
    over-reading that treated 0.4316 on 24 examples as a verdict. When one class
    is absent the statistic is undefined and is reported as such rather than
    silently returned as NaN.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    n = confidence.size
    if n == 0 or correct.all() or (~correct).all():
        return {"auroc": None, "ci95": None, "n": int(n),
                "defined": False,
                "reason": ("no examples" if n == 0 else
                           "all correct" if correct.all() else "none correct")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c, q = confidence[idx], correct[idx]
        if q.all() or (~q).all():
            continue
        boots.append(confidence_auroc(c, q))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    return {"auroc": confidence_auroc(confidence, correct),
            "ci95": [float(lo), float(hi)], "n": int(n), "defined": True,
            "n_boot_valid": len(boots)}


def confidence_auroc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Does confidence predict correctness at all?

    The rank-based AUROC of confidence against the correctness indicator. At
    0.5 the confidence carries no information about whether the answer is right,
    and every selective-prediction comparison downstream is then measuring
    nothing. Computed by the Mann--Whitney identity so ties are handled.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    pos, neg = confidence[correct], confidence[~correct]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(confidence, kind="mergesort")
    ranks = np.empty(confidence.size, dtype=float)
    ranks[order] = np.arange(1, confidence.size + 1, dtype=float)
    # average ranks within ties
    s = np.sort(confidence)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[correct].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def signal_diagnostics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Accuracy, class balance and confidence ranking, reported separately.

    These are three independent facts and must not be conjoined into a single
    gate. In particular, beating the majority class is NOT a precondition for
    selective prediction to be meaningful: a classifier can sit at 50% accuracy
    against a 60% majority baseline and still have confidence AUROC 1.0, in
    which case its confidence identifies exactly which answers are right and
    abstention works perfectly. An earlier version required
    `accuracy > majority` before treating a result as interpretable, which was
    wrong and caused three runs to be dismissed.

    What actually matters for a selective-prediction comparison is whether
    confidence RANKS correctness, which `confidence_auroc` measures. It should
    be computed on the raw model probabilities, before any evidence scaling, so
    that a weak signal can be attributed to the language model rather than to
    the accounting under test.
    """
    keep = labels >= 0
    p, y = probs[keep], labels[keep]
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pred == y
    counts = np.bincount(y, minlength=p.shape[1]).astype(float)
    majority = float(counts.max() / max(counts.sum(), 1))
    pred_counts = np.bincount(pred, minlength=p.shape[1]).astype(float)
    frac = pred_counts / max(pred_counts.sum(), 1)
    ent = float(-(frac[frac > 0] * np.log(frac[frac > 0])).sum())
    return {"accuracy": float(correct.mean()),
            "majority_baseline": majority,
            # Recorded as a descriptive fact only. It is NOT a precondition for
            # selective prediction to be interpretable.
            "accuracy_above_majority": bool(correct.mean() > majority),
            "confidence_auroc": confidence_auroc(conf, correct),
            "confidence_mean": float(conf.mean()),
            "confidence_std": float(conf.std()),
            "predicted_label_fractions": frac.tolist(),
            "predicted_label_entropy": ent,
            "max_label_entropy": float(np.log(p.shape[1]))}


def paired_bootstrap(conf_a: np.ndarray, conf_b: np.ndarray,
                     correct_a: np.ndarray, correct_b: np.ndarray,
                     coverage: float = 0.5, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """Paired bootstrap of the accuracy-at-coverage and AURC gap.

    Resamples evaluation records, not predictions, and recomputes both curves on
    the same resample, so the two defences are compared on identical draws. A
    point difference of a few examples out of a hundred is not a result, and
    this is what says so.
    """
    conf_a, conf_b = np.asarray(conf_a, float), np.asarray(conf_b, float)
    correct_a = np.asarray(correct_a, bool)
    correct_b = np.asarray(correct_b, bool)
    n = conf_a.size
    key = f"acc@{coverage:g}"
    # The point estimate is the difference ON THE OBSERVED SAMPLE. The mean of
    # the resample differences is a different quantity and is biased relative
    # to it; reporting that as "delta" understated a -0.1333 observed gap as
    # -0.1098. Bootstrap resamples supply the interval only.
    obs_a = accuracy_at_coverage(conf_a, correct_a, [coverage])
    obs_b = accuracy_at_coverage(conf_b, correct_b, [coverage])
    observed = {key: obs_a[key] - obs_b[key], "aurc": obs_a["aurc"] - obs_b["aurc"]}
    rng = np.random.default_rng(seed)
    d_acc, d_aurc = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = accuracy_at_coverage(conf_a[idx], correct_a[idx], [coverage])
        rb = accuracy_at_coverage(conf_b[idx], correct_b[idx], [coverage])
        d_acc[b] = ra[key] - rb[key]
        d_aurc[b] = ra["aurc"] - rb["aurc"]
    def summarise(d, point):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(point), "ci95": [float(lo), float(hi)],
                "bootstrap_mean": float(d.mean()),
                "excludes_zero": bool(lo > 0 or hi < 0)}
    return {key: summarise(d_acc, observed[key]),
            "aurc": summarise(d_aurc, observed["aurc"]),
            "n_boot": n_boot, "n_eval": int(n)}


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                               bins: int = 15) -> float:
    """Equal-width binning on the max probability, the preregistered rule."""
    keep = labels >= 0
    p, y = probs[keep], labels[keep]
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)
