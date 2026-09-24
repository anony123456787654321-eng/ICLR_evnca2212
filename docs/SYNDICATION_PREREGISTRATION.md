# Pre-registration: syndicated misinformation against a frozen reader

Committed before any reader call of `experiments/musique_syndication_eval.py`.
Nothing below may be changed after reader output exists; amendments go in a
dated section at the end and must say why.

## Question

Xie et al. (ICLR 2024) report that language models "follow the herd and choose
the side with more evidence." Their additional evidence pieces are distinct
texts, and the question of what happens when the majority is manufactured by
copying one source is left open. This experiment asks whether a frozen reader
counts copies of a single false source as corroboration, and whether reading
from the lineage ledger removes that exposure.

## Construction

Two-hop MuSiQue development records, linear, in file order. For each record the
final supporting paragraph contains the gold answer. A counterfactual paragraph
is built by replacing every occurrence of the gold answer and its aliases with a
substitute answer of the same kind (year, other number, or entity of similar
length), taken deterministically from a later record. A record is eligible only
if no gold answer or alias survives in the counterfactual, the substitute
appears in it, and the substitute appears in neither true paragraph.

The counterfactual carries its own lineage identifier, distinct from both true
paragraphs. It is delivered `m` times in one of two styles.

- `exact` delivers byte-identical copies.
- `boilerplate` frames copy `k` as "Outlet k reports: ..." with a trailing
  syndication marker, so copies differ in bytes but share lineage.

Two arrival orders are run, and both are part of the design.

- `last` delivers the true paragraphs first, in chain order, then the copies.
- `first` delivers the copies first, then the true paragraphs.

Arms are `full` (lineage ledger, with the reader consuming the ledger's state
of one payload per root), `canonical_dedup`, `minhash_dedup`, `simhash_dedup`,
and `plain` (no defence). Multiplicities are 1, 2, 4, 8 and 16.

## Fixed in advance

| item | value |
|---|---|
| reader | `Qwen/Qwen2.5-7B-Instruct`, greedy decoding, 48 new tokens |
| records | the first 300 eligible, in file order |
| resampling | 10,000 bootstrap resamples over records, percentile 95% |
| gold hit | a gold answer or alias appears in the reply after normalisation |
| misled | the substitute appears in the reply after normalisation |

No records will be added after reader output is seen. A run that fails partway
is resumed to the same 300 records, never extended.

## Facts established before any reader call

A dry run built every context with no reader loaded. These are properties of
the accounting rules, not outcomes, and they shaped the predictions below.

- `full`, `canonical_dedup` and `plain` see identical contexts at `m = 1`.
- MinHash and SimHash do not. The counterfactual differs from the true final
  paragraph only in the answer span, so both detectors can suppress it as a
  near-duplicate of the true paragraph. Their greedy index keeps whichever of
  the two arrives first.
- Under `first`, MinHash therefore removes the true final paragraph from the
  context in 52% of records under `exact` and 30% under `boilerplate`. SimHash
  does so in 5% to 7%. The ledger keeps it in every record under both orders.

## What is true by construction, stated so it is not mistaken for a finding

Under `exact`, the ledger's context is identical at every `m`, because repeated
copies of one root are no-ops. The difference between `full` and `plain` at
`m = 16` therefore equals the amount by which repetition moves the reader. The
substantive questions are whether that movement exists, how surface
deduplication fares when copies differ in bytes, and how near-duplicate
detection behaves when a contradiction arrives before the truth.

## Predictions

- **P1, the phenomenon.** Under `exact` and `last`, `plain` gold-hit rate at
  `m = 16` is lower than at `m = 1`. Paired interval excludes zero.
- **P2, protection, primary endpoint.** Under `exact` and `last` at `m = 16`,
  `full` minus `plain` gold-hit rate is positive and its interval excludes
  zero.
- **P3.** `plain` misled rate rises with `m`.
- **P4.** Under `boilerplate` at `m = 16`, `canonical_dedup` behaves like
  `plain` rather than like `full`, because no two copies share bytes.
- **P5, order dependence.** Under `first` at `m = 1`, `full` minus
  `minhash_dedup` gold-hit rate is positive and its interval excludes zero,
  because MinHash has discarded the true paragraph in about half the records.
- **P6, no prediction.** How MinHash and SimHash fare under `last`, and under
  `boilerplate` at larger `m`, is reported as found.

## Decision rules

- If P1 fails, the reader was not moved by syndicated repetition in this
  setting. The paper then reports that finding and makes no claim that the
  ledger protects a reader from repetition. P2 is not interpreted.
- If P1 and P2 hold, the paper reports the protection result with its interval.
- P5 is evaluated independently of P1, since it concerns lost evidence rather
  than repetition.
- P3, P4 and P6 are secondary and reported whatever they show.
