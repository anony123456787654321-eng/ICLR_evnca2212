# Pre-registration: directional source credit

Committed before either experiment below produced any result. Nothing here may
be changed after an outcome exists; amendments go in a dated section at the end
and must say why.

## Question

Lineage accounting counts each lineage-distinct source once. Every scalar form
of that idea, whether a source count (truth discovery, grouping by document
identifier) or a set of source identifiers (pedigree codes), credits a source
with support in every direction, including directions its observation never
constrained. The ledger instead credits source `r` with its own precision
`Lambda_r`, so support accumulates only along the directions each source
constrains. This document asks whether keeping credit directional matters, in a
learned model with known ground truth and on real retrieval data.

## Experiment A2: synthetic, learned message passing

Model and task are the synthetic evidence graph of `experiments/
architecture_matrix.py` (path topology, 8 cells, 3 roots, 4-dimensional latent,
8 message steps, 4000 training iterations, batch 16, hidden 64). Each model is
trained on clean batches and evaluated on held-out batches.

Observation rank is varied: `obs_dim` in {1, 2, 4}. With rank-1 observations
and three roots, at least one latent direction receives no observation at all.

Arms, all sharing one content path and differing only in how credit becomes a
precision matrix:

- `ec_matrix`: lineage ledger, `Lambda = prior I + sum_r credit_r Lambda_r`.
  This is the method.
- `ec_scalar`: lineage ledger, credit made isotropic with the same trace,
  `Lambda = prior I + sum_r credit_r (tr(Lambda_r) / d) I`. Same total
  information, no direction.
- `plain_matrix`, `plain_scalar`: the same two forms without lineage, so every
  arrival adds credit.

Conditions: clean, and every root delivered 16 times (`duplicate_batch`).
Seeds 0 to 4. Primary metric: Gaussian negative log-likelihood of the true
latent (`gaussian_nll`), mean over cells. Secondary: 90% coverage, RMSE.
Intervals are paired t intervals over seeds.

What is true by construction, stated so it is not mistaken for a finding: the
precision is assembled analytically from credit, so the scalar arms are
isotropic by definition. What is measured is how large the resulting
calibration penalty is once the mean is learned, and whether it depends on the
observation rank as the directional argument says it should.

- **S1, primary.** At `obs_dim = 1`, clean: `ec_matrix` NLL is lower than
  `ec_scalar` NLL, paired interval excluding zero.
- **S2.** At `obs_dim = 1`, 16x duplicated: `ec_matrix` NLL is lower than
  `plain_matrix` NLL, paired interval excluding zero.
- **S3, control.** The `ec_scalar` minus `ec_matrix` NLL gap at `obs_dim = 4`
  is smaller than at `obs_dim = 1` (point estimates).

## Experiment A3: real data, MuSiQue retrieval sufficiency

Data: every answerable record of `musique_ans_v1.0_dev.jsonl`, all hop counts,
file order. Each record carries 20 paragraphs, of which the supporting ones are
annotated.

Retrieval: paragraphs and queries are encoded with BGE `bge-base-en-v1.5`,
L2-normalised, with the retrieval instruction applied to queries only. The
retrieved set is the top `k = 5` of the record's 20 paragraphs by cosine
similarity to the question.

Label: a retrieved set is **sufficient** when it contains every supporting
paragraph. The label is the only place support annotations enter.

Requirement directions: for each decomposition step, the step text with every
`#n` reference removed and `>>` replaced by a space, encoded as a query. No
intermediate answer, no `paragraph_support_idx`, and no support label enters
any score. The decomposition stands in for a query decomposer, as used in
multi-hop retrieval, and this is disclosed.

Scores, computed over the lineage-distinct sources of the retrieved set, where
source `r` has embedding `e_r` and contributes rank-one credit `e_r e_r^T`:

- `s_h = sum_r (d_h . e_r)^2`, the credited support along requirement
  direction `d_h`.
- **Directional** `D = min_h s_h`, the support in the weakest required
  direction. This is the method.
- **Scalar** `S = mean_h s_h`, the same support summed without regard to
  direction. Identical inputs; only the aggregation differs.
- Reported baselines, standard retrieval confidences: sum and max of the
  question-to-passage cosines over the retrieved set.

Lineage condition. In the duplicated condition, four copies of the top-ranked
paragraph are added to the record's pool, each carrying that paragraph's
lineage identifier, and retrieval is repeated, so copies can crowd other
paragraphs out of the top five. Lineage-aware scores count the copies once;
provenance-free scores count every retrieved arrival. Sufficiency is judged on
the retrieved content, so copies never make a set sufficient.

Primary metric: AUROC of each score for predicting sufficiency. Intervals: 10,000
paired bootstrap resamples over records, percentile 95%.

- **R1, primary.** Clean: AUROC of `D` exceeds AUROC of `S`, interval excluding
  zero.
- **R2.** Duplicated: AUROC of lineage-aware `D` exceeds AUROC of
  provenance-free `S`, interval excluding zero. This compares the method with
  the standard of counting whatever was retrieved.
- **R3, no prediction.** The remaining cells of the lineage-by-aggregation
  design and the two retrieval baselines are reported as found.

## Decision rules

- If S1 fails, the synthetic experiment is reported as not supporting
  directional credit, and no synthetic directional claim is made.
- If R1 fails, no real-data directional claim is made; R2 is still reported.
- Every outcome, including failures, appears in the paper or its appendix.

## Amendment, 24 Sep 2026, before any A3 run

Writing the A3 script showed, before any score or label was computed, that an
exact copy ties with its original under cosine retrieval. Four copies of the
top-ranked paragraph would therefore take four of the five slots, every
duplicated set would contain one distinct paragraph, the sufficiency label
would be constant, and AUROC would be undefined. The duplicated condition is
therefore changed so that the four copies are appended to the retrieved top
five rather than competing for its slots, which is how the syndication
experiment delivers copies to a reader. The label is unchanged by the copies,
since they add no content. Nothing else in A3 changes. No A3 output existed
when this amendment was written.

## Amendment 2, 24 Sep 2026, before any A2 or A3 output

An independent audit of this document, made before any A2 or A3 output
existed, found two ways the tests as written could pass for the wrong reason.
Both are closed here by making the tests stricter. The A3 run in progress was
stopped before it wrote anything, and no A2 model had finished training.

A2. The scalar arm `ec_scalar` is isotropic with a fixed scale, so at
`obs_dim = 1` it assigns precision to directions no root observed, and S1 is
close to guaranteed. A fifth arm, `ec_scalar_fit`, is added. It is the same
isotropic credit multiplied by a learnable global scale trained jointly with
the model, so it can learn how confident to be but not in which direction.
**S1 and S3 now compare `ec_matrix` with `ec_scalar_fit`.** The contrast
with the fixed-scale `ec_scalar` is still reported.

A3. The decomposition has one step per supporting paragraph, so a minimum over
steps could track the number of hops, and hop count alone predicts whether
five retrieved paragraphs contain every support. Three changes follow.

- **R1 is evaluated within hop-count strata.** The stratified AUROC is the
  size-weighted mean of the AUROCs within 2-, 3- and 4-hop records, and the
  bootstrap resamples records within strata. R1 holds when the stratified
  `D` minus `S` interval excludes zero. The unstratified contrast is still
  reported.
- Added baselines, reported without prediction: hop count alone (fewer hops
  scores higher), `best_min = min_h max_r cos(d_h, e_r)`, and
  `mean_best = mean_h max_r cos(d_h, e_r)`. The last two use the decomposition
  but do not accumulate credit across sources.
- R2 is also reported stratified.

The decomposition is an oracle for the query structure, and the paper says so.
