#!/usr/bin/env bash
# Re-chunking sweep for the learned two-hop memory, with the whole paragraph in
# the stream.
#
#   bash scripts/dgx_rechunk.sh                          rechunk (paragraph + windows)
#   INTERVENTION=fragments bash scripts/dgx_rechunk.sh   windows only, paragraph withheld
#
# The frozen `overlap` sweep (results/musique_dup_iclr_fusion_h2) cut at most
# three partial windows from a paragraph and never delivered the paragraph
# itself, so the ledger could only hold a partial window. The `rechunk`
# intervention delivers the paragraph once in full and then distinct windows
# cut from it, sixteen renderings in all at sixteenfold. Each window ranks
# below the paragraph by the words it omits, so the ledger holds the whole
# paragraph, while exact hashing sees sixteen distinct surfaces.
#
# Out-of-distribution evaluation of the clean-trained checkpoints only. Nothing
# is trained. Output lands in a fresh tree, results/musique_rechunk_h2, so no
# frozen result is touched. The embedding cache is copied, not shared, so the
# clean arrivals are encoded exactly as in the frozen run, and the script
# checks that the clean rows reproduce it.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG="logs/dgx_rechunk_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

INTERVENTION="${INTERVENTION:-rechunk}"
OUT="${OUT:-results/musique_${INTERVENTION}_h2}"
CKPT="${CKPT:-results/architecture/transport}"
CACHE_SRC="${CACHE_SRC:-results/musique/bge_cache.npz}"
FROZEN="results/musique_dup_iclr_fusion_h2/eval"
VARIANTS="full,plain,canonical_dedup,minhash_dedup,simhash_dedup,embed_dedup,dempster_fusion,covariance_intersection"
DEVICE=cpu
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" && DEVICE=cuda

echo "== commit $(git rev-parse --short HEAD), device $DEVICE, intervention $INTERVENTION, out $OUT"
DEV=data/raw/musique/data/musique_ans_v1.0_dev.jsonl
if [ ! -f "$DEV" ]; then python scripts/musique_data.py restore; fi
for v in full plain; do
  for s in 0 1 2 3 4; do
    test -f "$CKPT/${v}_s$s/ckpt.pt" || { echo "FATAL: missing $CKPT/${v}_s$s/ckpt.pt"; exit 2; }
  done
done
python -m pytest -q tests/test_musique_duplication.py tests/test_near_duplicate.py

mkdir -p "$OUT"
if [ -s "$CACHE_SRC" ] && [ ! -s "$OUT/bge_cache.npz" ]; then
  cp "$CACHE_SRC" "$OUT/bge_cache.npz"
fi

echo "== smoke, 32 records"
python experiments/musique_duplication_eval.py --out "$OUT/smoke" \
  --checkpoints "$CKPT" --encoder bge --device "$DEVICE" \
  --cache "$OUT/bge_cache.npz" --variants full,plain --seeds 0 --hops 2 \
  --interventions "$INTERVENTION" --scopes all_hops --limit 32 --force
rm -rf "$OUT/smoke"

echo "== sweep: 8 arms x 5 seeds, two-hop, $INTERVENTION"
python experiments/musique_duplication_eval.py --out "$OUT/eval" \
  --checkpoints "$CKPT" --encoder bge --device "$DEVICE" \
  --cache "$OUT/bge_cache.npz" --variants "$VARIANTS" --seeds 0,1,2,3,4 \
  --hops 2 --interventions "$INTERVENTION" --scopes all_hops,one_hop --batch-size 64

n=$(ls "$OUT"/eval/*_s*.json | wc -l)
echo "cells: $n of 40"
[ "$n" -eq 40 ] || { echo "FATAL: incomplete sweep"; exit 1; }

echo "== summary and clean-row check against the frozen run"
python - "$OUT/eval" "$FROZEN" <<'PY'
import glob, json, sys, statistics as st
from pathlib import Path
new, frozen = Path(sys.argv[1]), Path(sys.argv[2])
acc = {}
for f in sorted(new.glob("*_s*.json")):
    d = json.loads(f.read_text())
    for r in d["rows"]:
        if r["multiplicity"] == 1 or r["scope"] == "all_hops":
            key = (d["variant"], r["multiplicity"])
            acc.setdefault(key, []).append(r["content"]["accuracy"])
            if r["multiplicity"] == 16:
                acc.setdefault((d["variant"], "credit16"), []).append(
                    r["credited_over_ceiling"])
for v in sorted({k[0] for k in acc}):
    cells = [f"x{m} {st.mean(acc[(v, m)]):.4f}" for m in (1, 2, 4, 8, 16)
             if (v, m) in acc]
    print(f"{v:>24}  " + "  ".join(cells)
          + f"  credit@16 {st.mean(acc[(v, 'credit16')]):.3f}")
mismatch = 0
for f in sorted(new.glob("*_s*.json")):
    g = frozen / f.name
    if not g.is_file():
        continue
    a = [r for r in json.loads(f.read_text())["rows"] if r["multiplicity"] == 1][0]
    b = [r for r in json.loads(g.read_text())["rows"] if r["multiplicity"] == 1][0]
    if abs(a["content"]["accuracy"] - b["content"]["accuracy"]) > 1e-9:
        mismatch += 1
        print("clean row differs from frozen run:", f.name,
              a["content"]["accuracy"], b["content"]["accuracy"])
print("clean rows reproduce the frozen run" if mismatch == 0
      else f"WARNING: {mismatch} clean rows differ from the frozen run")
PY

echo "== push raw outputs"
git add -f "$OUT"/eval "$LOG"
git commit -m "Record the $INTERVENTION re-chunking sweep"
git push
echo "== done"
