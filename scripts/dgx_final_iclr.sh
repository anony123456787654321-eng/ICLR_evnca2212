#!/usr/bin/env bash
# Final confirmatory DGX sequence: Semantic-Sybil audit, then the MuSiQue
# transport sweep, then a commit of the two report files.
#
# Everything is appended to logs/final_iclr_jobs.log. Nothing is retrained and
# no model, dataset, threshold, candidate corpus or evaluation metric changes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="$ROOT/logs/final_iclr_jobs.log"
exec > >(tee -a "$LOG") 2>&1

BRANCH="codex/averitec-rag"
SYBIL_OUT="results/semantic_sybil"
MUSIQUE_OUT="results/musique/dgx"

stage() { echo; echo "===== [$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $* ====="; }
fail()  { echo "FATAL: $*" >&2; exit 2; }

stage "context"
git rev-parse HEAD
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The Semantic-Sybil embedding cache is expensive and is deliberately reused.
if [ -s "$SYBIL_OUT/bge_cache.npz" ]; then
  echo "[cache] reusing $SYBIL_OUT/bge_cache.npz ($(stat -c%s "$SYBIL_OUT/bge_cache.npz" 2>/dev/null || stat -f%z "$SYBIL_OUT/bge_cache.npz") bytes)"
else
  echo "[cache] no existing embedding cache; it will be built once"
fi

stage "musique data: restore and validate"
# Handled before any expensive stage so a missing dataset fails in seconds.
if ! python scripts/musique_data.py verify >/dev/null 2>&1; then
  python scripts/musique_data.py restore
fi
python scripts/musique_data.py verify

stage "checkpoints: restore and validate"
if ! python scripts/gate_c_checkpoints.py verify --run results/gate_c >/dev/null 2>&1; then
  python scripts/gate_c_checkpoints.py restore --run results/gate_c
fi
python scripts/gate_c_checkpoints.py verify --run results/gate_c

# Resumable: a completed audit over the frozen 200-candidate corpus is reused
# rather than recomputed, so a failure in a later stage costs only that stage.
if python -c 'import json,sys; sys.exit(0 if json.load(open("'"$SYBIL_OUT"'/report.json"))["population"]==200 else 1)' 2>/dev/null; then
  stage "semantic sybil: audit already complete, reusing $SYBIL_OUT/report.json"
else
  stage "semantic sybil: env"
  bash scripts/dgx_semantic_sybil.sh env
  stage "semantic sybil: audit"
  bash scripts/dgx_semantic_sybil.sh audit
fi
# collect always runs: it derives summary.log from report.json and costs nothing.
stage "semantic sybil: collect"
bash scripts/dgx_semantic_sybil.sh collect

stage "musique: env"
bash scripts/dgx_musique.sh env
stage "musique: dry"
bash scripts/dgx_musique.sh dry
stage "musique: sweep"
bash scripts/dgx_musique.sh sweep
stage "musique: collect"
bash scripts/dgx_musique.sh collect

stage "publish reports"
REPORTS=(
  "$SYBIL_OUT/report.json"
  "$SYBIL_OUT/summary.log"
  "$MUSIQUE_OUT/runs/report.json"
  "$MUSIQUE_OUT/report.log"
)
for f in "${REPORTS[@]}"; do
  [ -s "$f" ] || fail "expected report missing: $f"
done
# results/ and *.log are gitignored; report files are force-added, as with the
# earlier confirmatory runs already tracked under results/.
git add -f "${REPORTS[@]}"
if git diff --cached --quiet; then
  echo "[publish] reports unchanged; nothing to commit"
else
  git -c user.name='Anonymous' -c user.email='anonymous@anonymous.invalid' \
    commit -q -m "Record confirmatory Semantic-Sybil and MuSiQue DGX reports

Outputs of scripts/dgx_final_iclr.sh on the DGX: the Semantic-Sybil audit
over the frozen 200-candidate corpus using the restored Gate C ec_exact
checkpoints, and the MuSiQue transport sweep over four variants and five
seeds. Reports only; no model, dataset, threshold, candidate corpus or
evaluation metric changed, and nothing retrained."
  git log --oneline -1
  # A credential prompt must not hang a nohup'd run or discard the commit.
  if ! GIT_TERMINAL_PROMPT=0 git push origin "$BRANCH"; then
    echo "[publish] WARNING: push failed (likely credentials). The commit is"
    echo "[publish] safe locally; run 'git push origin $BRANCH' interactively."
  fi
fi

stage "done"
echo "log:      $LOG"
echo "bundles:  $ROOT/$SYBIL_OUT.tgz  $ROOT/$MUSIQUE_OUT.tgz"
