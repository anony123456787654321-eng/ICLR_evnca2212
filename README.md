# Repetition Is Not Corroboration: Counting Sources Instead of Copies in Neural Memory

Anonymous code and data release accompanying the submission of the same name.

This tree contains the implementation, the committed raw outputs that the
paper's tables and figures are generated from, the pre-registration documents,
and the test suite that guards the structural properties the paper asserts.

## Layout

    ecnca/          the library: ledger, sketch, fusion and backbone models
    experiments/    the runs that produce raw outputs
    analysis/       the scripts that turn raw outputs into tables and figures
    results/        the committed raw outputs the manuscript reads
    docs/           pre-registration documents, committed before the runs
    paper/          manuscript source, generated table bodies and figures
    tests/          property tests for the ledger, sketch and detectors

## Regenerating the manuscript's tables

Everything below reads committed outputs and runs on CPU:

    python analysis/syndication_analysis.py     # pre-registered syndication verdict
    python analysis/story_tables.py             # syndication tables and Figure 1
    python analysis/directional_analysis.py     # directional credit tables and verdict
    python analysis/learned_memory_table.py     # learned memory, all three regimes
    python analysis/axiom_predictions.py        # accounting-property table
    python analysis/conservation_figures.py     # backbone figure

## Re-running the experiments

These need a GPU. The syndication experiment with a frozen reader:

    python experiments/musique_syndication_eval.py --position last \
        --out results/musique_syndication/last
    python experiments/musique_syndication_eval.py --position first \
        --out results/musique_syndication/first

Directional credit, synthetic and on MuSiQue retrieval:

    for arm in ec_matrix ec_scalar_fit ec_scalar plain_matrix plain_scalar; do
      python experiments/directional_synthetic.py --arms $arm \
          --out results/directional_synthetic/$arm
    done
    python experiments/directional_musique.py --device cuda

The learned multi-hop memory: train the clean two-hop checkpoints, then
evaluate them under exact redelivery, re-chunking with the paragraph among its
windows, and fragments only. Nothing is trained on duplicated streams.

    python experiments/gate5_musique.py --out results/architecture/transport \
        --encoder bge --device cuda --iters 10000 --batch-size 128 \
        --seeds 0,1,2,3,4 --variants full,plain
    ARMS=full,plain,canonical_dedup,minhash_dedup,simhash_dedup,embed_dedup,dempster_fusion,covariance_intersection
    python experiments/musique_duplication_eval.py --hops 2 --encoder bge \
        --device cuda --checkpoints results/architecture/transport \
        --variants $ARMS --interventions exact,overlap \
        --out results/musique_dup_iclr_fusion_h2/eval
    for iv in rechunk fragments; do
      python experiments/musique_duplication_eval.py --hops 2 --encoder bge \
          --device cuda --checkpoints results/architecture/transport \
          --variants $ARMS --interventions $iv --scopes all_hops,one_hop \
          --out results/musique_${iv}_h2/eval
    done

## Pre-registration

`docs/SYNDICATION_PREREGISTRATION.md` and `docs/DIRECTIONAL_PREREGISTRATION.md`
fix the predictions, endpoints and decision rules of the reader and
directional experiments, and `docs/FRAGMENTS_PREDICTION.md` fixes those of the
fragments regime. Each was committed before its experiment produced output, and
the analysis scripts apply the rules mechanically.

## Tests

    python -m pytest tests -q

## Data

MuSiQue is used under CC BY 4.0. AVeriTeC is used under its released licence
and is not redistributed here; the official knowledge store is used in place of
live web search. Encoders and readers are public checkpoints at pinned
revisions, and none is fine-tuned.
