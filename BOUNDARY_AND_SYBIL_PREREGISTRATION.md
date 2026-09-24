# Boundary placement and semantic-Sybil preregistration

Frozen before calculating dataset placement or running the semantic-paraphrase
attack. These analyses may explain an existing result or quantify a limitation;
they may not redefine the Tier-1 benefit surface.

## Dataset placement on the dynamic-sector boundary

The synthetic boundary axes are the number of simultaneously valid target
modes and their separation relative to within-mode noise.

For AVeriTeC, target cardinality is one because each claim has one adjudicated
verdict. Separation is not defined for a unimodal target and will be plotted as
`N/A`, not assigned an arbitrary coordinate.

For AmbigNQ, cardinality is the number of interpretations in the official
annotation. Each interpretation centroid is the mean frozen-BGE embedding of
its answer aliases. Between-mode distance is the RMS pairwise Euclidean
distance between centroids. Within-mode noise is the pooled RMS distance of
answer aliases from their interpretation centroid, estimated globally from
all multi-alias interpretations when an example has singleton aliases. The
reported separation ratio is between-mode distance divided by this frozen
global noise estimate. No model prediction or Gate-4G outcome enters either
descriptor.

The plot will mark AmbigNQ only after reporting the descriptor distribution and
will explicitly label the text-embedding ratio as an analogue of, not an exact
unit conversion to, the synthetic latent-space separation axis.

## Semantic-paraphrase Sybil attack

Population: the first 200 AVeriTeC retrieved-development claims with at least
one passage, in deterministic record order. The attack source is the highest
ranked passage. Every paraphrase keeps one declared true source but is assigned
a distinct URL/root identity.

Generation uses `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`, with greedy decoding, eight
explicitly numbered rewrite instructions, temperature disabled, and a maximum
of 128 new tokens. The generator is used only to construct an attack; it is
never trained or scored as part of EC-NCA. A candidate is retained
only when:

1. frozen BGE cosine similarity to the source is at least 0.80;
2. the train-frozen surface-family score is below its 0.87 merge threshold;
3. length is between 0.5 and 1.5 times source length; and
4. normalised text is not identical to another retained candidate.

Pre-outcome validity amendment (commit following manual inspection of generator
text, before any EC model score): lexical/embedding similarity can retain a
rewrite that adds or reverses a fact. Each candidate must therefore also receive
the `entailment` argmax in both directions from
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at revision
`6f5cf0a2b59cabb106aca4c287eed12e357e90eb`. All rejected candidates and their
two label/probability vectors remain in the manifest. This filter validates the
attack content; it does not use EC predictions, labels, or credited evidence.

Claims with fewer than four retained paraphrases are reported and excluded
from the primary paired attack, never silently replaced. Attack multiplicities
are 1, 2, 4 and 8, capped by the retained count.

Frozen Gate-C `ec_exact` models and the frozen BGE cache are reused without
training. The primary quantity is credited evidence relative to the 1-copy
source baseline. Secondary quantities are NLL, prediction flip rate and
confidence change. Comparisons are URL lineage, canonical URL deduplication,
the confirmed surface-family rule, and an exploratory semantic-family rule.

The semantic-family cosine threshold is selected on AVeriTeC train only as the
highest-recall threshold on exact/surface mirror positives subject to at most
0.5% false links among cross-root, nonidentical within-claim pairs. It is frozen
before dev attack evaluation.

This audit is allowed to fail. If independently worded paraphrases inflate
credit, the paper will report the measured factor and will not describe source
identity as semantic independence.
