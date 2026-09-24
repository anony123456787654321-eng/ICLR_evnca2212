# Fragments-only re-chunking: prediction, 24 Sep 2026

Committed with the code and before any output of this regime exists.

**Regime.** `fragments` in `ecnca/real/musique_duplication.py`: at multiplicity
m, every hop of a two-hop chain arrives as m distinct contiguous windows of its
supporting paragraph (the windows of `rechunk` without the paragraph itself).
All windows carry the paragraph's lineage identifier. Same checkpoints, seeds,
readout, arms and statistics as the `rechunk` sweep
(`results/musique_rechunk_h2`), run by `INTERVENTION=fragments bash
scripts/dgx_rechunk.sh`.

**Purpose.** `rechunk` measures the case where a complete rendering of the
source is among its copies. This regime measures the case where none is, so the
ledger can hold only a fragment. Together they separate what lineage does for
credit from what it does for accuracy.

**Predictions at x16, content readout, paired t over five seeds.**

- F1. Credit: the ledger's credited evidence is exactly 1.000 of the
  lineage-distinct ceiling; no defence is 16.000.
- F2. Accuracy: the ledger minus no defence difference in top-1 accuracy has an
  interval that includes zero. The ledger holds a fragment and is not expected
  to beat a reader that sees every fragment.

**Reporting rule.** Whatever the outcome, the regime is reported in the main
text beside `rechunk`, with its interval. If F2 fails in either direction, the
text says which way. The earlier three-window run
(`results/musique_dup_iclr_fusion_h2`, `overlap`) stays reported in the
appendix.
