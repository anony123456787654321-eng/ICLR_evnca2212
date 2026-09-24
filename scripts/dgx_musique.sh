#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGE="${1:-help}"
OUT="${OUT:-results/musique/dgx}"
ITERS="${ITERS:-10000}"
BATCH="${BATCH:-128}"

case "$STAGE" in
  data)
    python scripts/musique_data.py restore
    python scripts/musique_data.py verify
    ;;
  env)
    # Data FIRST. data/raw/ is gitignored, so a machine that only pulls has no
    # dataset; restoring and validating before the encoder download means a
    # missing dataset costs seconds rather than a failure at the sweep.
    if ! python scripts/musique_data.py verify >/dev/null 2>&1; then
      echo "[env] MuSiQue absent or invalid; restoring from tracked archives"
      python scripts/musique_data.py restore
    fi
    python scripts/musique_data.py verify
    python -c 'import transformers, sentence_transformers, sentencepiece; print(transformers.__version__, sentence_transformers.__version__)'
    python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())'
    python -m pytest -q tests/test_musique.py tests/test_musique_transport.py
    ;;
  dry)
    python experiments/gate5_musique.py --out "$OUT/dry" --encoder bge --device cuda \
      --iters 20 --batch-size 8 --variants full,filter_only --seeds 0
    ;;
  sweep)
    mkdir -p "$OUT"
    python experiments/gate5_musique.py --out "$OUT/runs" --encoder bge --device cuda \
      --iters "$ITERS" --batch-size "$BATCH" --variants full,filter_only,no_provenance,plain \
      --seeds 0,1,2,3,4
    ;;
  collect)
    python -c 'import json, pathlib; p=pathlib.Path("'"$OUT"'/runs/report.json"); d=json.loads(p.read_text()); assert len(d["rows"])==20; print(json.dumps(d, indent=2))' \
      | tee "$OUT/report.log"
    tar -czf "$OUT.tgz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
    ;;
  *)
    echo "usage: $0 data|env|dry|sweep|collect" >&2
    exit 2
    ;;
esac
