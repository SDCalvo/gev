#!/bin/bash
# The released Gev-E4B recipe, end to end and unattended. About 12 hours on an M3 Max (48 GB); resumable.
#
#   scripts/build_public_pool.py --n_per_source 2000 --seed 1 --out evals/public-pool-2k     (once; 20k public records)
#   scripts/build_delta_data.py --out evals/delta-v1                                          (once; targeted generated data)
#   DATES_OUT=evals/dates-v2 DATES_SEED=dates-v2-20260925 DATES_N=300 UNKNOWABLE_N=25 ASSERTION_N=400 scripts/build_date_data.py
#   scripts/build_delta_v2.py --out evals/delta-v2                                            (once; MCQ mix, paraphrase, rules)
#   scripts/recipe.sh [BASE] [OUT]        BASE defaults to google/gemma-4-E4B-it, OUT to runs/gev-e4b
#   READOUT_LAYERS=10,20 scripts/recipe.sh google/gemma-4-E2B-it runs/gev-e2b     (E2B has 35 layers: read 10, 20 and the last)
#
# Stages (markers in runs/logs/pipeline.status, logs in runs/logs/<out>-*.log):
#   base   1 epoch on decision-v7 + public-pool-2k + delta-v1 + delta-v2 (36k records): LoRA r=16 lr 1e-4, pointer head
#          lr 2e-4 with layer-normalized inputs reading layers 12, 24 and the final layer, cross-entropy + 0.5 Brier,
#          400 head-only warm-up steps, checkpoints every 100 steps (auto-resume on failure)
#   eval   decision-v7 development (in distribution) and transfer-v4 development (out of domain)
#   calib  temperature fitted on development rows, written into head.pt
#   delta  optional: --delta runs the delta-v1 and delta-v2 passes on top (lr 2e-5, 1 epoch each, replaying suite
#          records). On the instruction-tuned base they tie with the base run; on google/gemma-4-E4B they are worth
#          +5 to +9 points (docs/results.md).
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=${1:-google/gemma-4-E4B-it}; OUT=${2:-runs/gev-e4b}; NAME=$(basename "$OUT")
PY=.venv/bin/python; LOG=runs/logs; mkdir -p "$LOG"; STATUS=$LOG/pipeline.status
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
fail() { stage "FAILED: $*"; exit 1; }
REV=$($PY -c "import json,sys; print(json.load(open('evals/v7/decision-v7/manifest.json'))['base_revisions'].get('$BASE',''))")
[ -n "$REV" ] || REV=$($PY -c "from huggingface_hub import HfApi; print(HfApi().model_info('$BASE').sha)")
FLAGS="--base $BASE --base_revision $REV --weights_dtype bf16 --head_norm 1 --readout delimiter --readout_layers ${READOUT_LAYERS:-12,24} --brier_w 0.5 --accum 8"
[ -f evals/v8-data/train.jsonl ] || { mkdir -p evals/v8-data; cat evals/public-pool-2k/train.jsonl evals/delta-v1/train.jsonl > evals/v8-data/train.jsonl; }
TRAIN="$PY -m gev.train $FLAGS --data evals/v8-data/train.jsonl --suite evals/v7/decision-v7 --replay 12576 --epochs 1 --lr 1e-4 --head_lr 2e-4 --head_warmup_steps 400 --save_every 100 --out $OUT"

evaluate() {   # evaluate <run>: development + transfer benchmarks, calibration
  $PY -m gev.benchmark --run "$1" --suite evals/v7/decision-v7 --out "$1/development" > "$LOG/$(basename "$1")-dev.log" 2>&1 || fail "$1 development benchmark"
  $PY -m gev.benchmark --run "$1" --suite evals/v4/transfer-v4 --out "$1/transfer" > "$LOG/$(basename "$1")-transfer.log" 2>&1 || fail "$1 transfer benchmark"
  $PY scripts/calibrate_checkpoint.py --run "$1" --rows "$1/development/rows.json" --transfer "$1/transfer/rows.json" > "$LOG/$(basename "$1")-calibrate.log" 2>&1 || fail "$1 calibration"
}

if [ ! -f "$OUT/head.pt" ]; then
  if [ -d "$OUT/ckpt" ]; then stage "base: resuming $OUT"; $TRAIN --resume >> "$LOG/$NAME-train.log" 2>&1; else stage "base: training $OUT on $BASE"; $TRAIN > "$LOG/$NAME-train.log" 2>&1; fi
  for i in 1 2 3 4 5 6; do [ -f "$OUT/head.pt" ] && break; stage "base: exited without a final save, resuming (attempt $i)"; sleep 20; $TRAIN --resume >> "$LOG/$NAME-train.log" 2>&1; done
  [ -f "$OUT/head.pt" ] || fail "base training did not finish (see $LOG/$NAME-train.log)"
fi
stage "eval: $OUT"; evaluate "$OUT"; stage "eval: $OUT done"

if [ "${DELTA:-0}" = 1 ]; then
  prev=$OUT
  for d in delta-v1:2000 delta-v2:3000; do
    name=$OUT-${d%%:*}; stage "delta: ${d%%:*} on top of $prev -> $name"
    rm -rf "$name"
    $PY -m gev.train $FLAGS --data evals/${d%%:*}/train.jsonl --suite evals/v7/decision-v7 --replay ${d#*:} --init_from "$prev" --lr 2e-5 --head_lr 2e-5 --epochs 1 --save_every 200 --out "$name" > "$LOG/$(basename "$name")-train.log" 2>&1 || fail "delta ${d%%:*}"
    prev=$name
  done
  stage "eval: $prev"; evaluate "$prev"
  for s in development transfer; do $PY -m gev.compare --candidate "$prev/$s" --reference "$OUT/$s" --out "$prev/$s/vs-base.json" > /dev/null 2>&1 || true; done
  stage "eval: $prev done"
fi
stage "DONE: $OUT"
