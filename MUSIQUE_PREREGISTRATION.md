# MuSiQue real multi-hop preregistration

Frozen before loading MuSiQue outcomes or training a model. The benchmark is
MuSiQue-Answerable v1.0 (CC BY 4.0), selected because its official construction
filters examples that can be solved by disconnected reasoning and exposes the
constituent questions, answers, supporting paragraphs and composition graph.

## Population and split

We use examples whose official decomposition forms one connected linear chain:
each step after the first depends on the immediately preceding answer, every
step has a supporting paragraph, and the terminal decomposition answer matches
the official final answer after normalisation. Branched DAGs are reported but
excluded from the primary transport test because a single-message join does not
define their merge order.

Training uses official train examples with two or three hops. The primary
generalisation population is official dev examples with four hops, never used
for model selection. A deterministic train validation slice is selected by a
hash of the official example id. The official test split is untouched.

## Architecture and information audit

Frozen BGE encodes each sub-question, supporting paragraph and gold sub-answer.
A shared learned transition receives only the incoming message, the current
sub-question and that step's private supporting paragraph, and emits the next
message. The terminal answer selector receives only the final joined message
and a within-example candidate set of answer embeddings. It receives no full
question, hop index, future paragraph, future answer, persistent cell state or
example id. The same root identity and evidence ceiling travel through every
legitimate transformation.

`full` retains the latest version of that root; `filter_only` retains the first;
`no_provenance` and `plain` count occurrences. All learned variants have matched
parameter counts. An exact oracle copies each official sub-answer in sequence.

Before training, tests must establish: changing a downstream supporting
paragraph cannot change any upstream message; permuting downstream operators
changes the oracle terminal answer; the filter's terminal payload is unchanged
by every downstream step; the model readout has no alternative access to future
inputs; and exact redelivery changes neither the full payload nor its credit.

## Frozen primary outcomes

1. Four-hop terminal answer top-1 accuracy and MRR.
2. Terminal embedding cosine error, paired `full` versus `filter_only`.
3. Generalisation curve for two, three and unseen four hops.
4. Credited evidence / ceiling under 1, 2, 4, 8 and 16 exact deliveries and on
   a cycle formed by redelivering earlier message versions.
5. Prediction change under a legitimate next transformation.
6. Prediction and precision change under exact redelivery.

The refinement claim requires a positive paired full-minus-filter difference
on unseen four-hop top-1 accuracy with a 95% interval above zero, while the
upper 95% confidence bound of full evidence/ceiling remains at most 1.05. A
null or negative result is retained. CPU work is limited to loader/oracle tests
and a 100-step smoke run; the five-seed confirmatory run is a separate DGX stage.
