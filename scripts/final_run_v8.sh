#!/bin/bash
# The stages after a passed pilot gate: full training (resumable), evaluation, calibration. Unattended (nohup this).
# Training checkpoints every 100 optimizer steps into runs/e4b-v8/ckpt; a crash resumes from the last checkpoint, up
# to MAX_RETRIES times. Stage markers go to runs/logs/pipeline.status; each stage logs to runs/logs/full-*.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/logs; mkdir -p "$LOG"
STATUS="$LOG/pipeline.status"
MAX_RETRIES=${MAX_RETRIES:-6}
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
fail() { stage "FAILED: $*"; exit 1; }
TRAIN="$PY -m systemone.train --data evals/v8-data/train.jsonl --suite evals/v7/decision-v7 --replay 12576 --epochs 1 --weights_dtype bf16 --lr 1e-4 --head_lr 2e-4 --head_norm 1 --head_warmup_steps 400 --readout delimiter --readout_layers 12,24 --brier_w 0.5 --accum 8 --save_every 100 --out runs/e4b-v8"

attempt=0
if [ -f runs/e4b-v8/head.pt ]; then
  stage "full: runs/e4b-v8 already trained, skipping to evaluation"
else
  if [ -d runs/e4b-v8/ckpt ]; then stage "full: resuming runs/e4b-v8 from its checkpoint"; $TRAIN --resume >> "$LOG/v8-train.log" 2>&1
  else stage "full: training runs/e4b-v8 (Kev-4B recipe, 2 epochs, checkpoint every 100 steps)"; $TRAIN > "$LOG/v8-train.log" 2>&1; fi
  while [ ! -f runs/e4b-v8/head.pt ] && [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    stage "full: training exited without a final save (attempt $attempt of $MAX_RETRIES), resuming from the checkpoint"
    tail -3 "$LOG/v8-train.log" >> "$STATUS"
    [ -d runs/e4b-v8/ckpt ] || fail "no checkpoint to resume from"
    sleep 20
    $TRAIN --resume >> "$LOG/v8-train.log" 2>&1
  done
  [ -f runs/e4b-v8/head.pt ] || fail "full training did not finish after $MAX_RETRIES resumes (see $LOG/v8-train.log)"
fi

stage "eval: decision-v7 development (in distribution)"
$PY -m systemone.benchmark --run runs/e4b-v8 --suite evals/v7/decision-v7 --out runs/e4b-v8/development > "$LOG/v8-dev.log" 2>&1 || fail "development benchmark"
stage "eval: transfer-v4 development (out of domain)"
$PY -m systemone.benchmark --run runs/e4b-v8 --suite evals/v4/transfer-v4 --out runs/e4b-v8/transfer > "$LOG/v8-transfer.log" 2>&1 || fail "transfer benchmark"
stage "calibrate: fitting the temperature on development rows"
$PY scripts/calibrate_checkpoint.py --run runs/e4b-v8 --rows runs/e4b-v8/development/rows.json --transfer runs/e4b-v8/transfer/rows.json > "$LOG/v8-calibrate.log" 2>&1 || fail "calibration"
stage "DONE: runs/e4b-v8 trained, evaluated and calibrated"
stage "compare: e4b-v8 vs e4b-v7 and e4b-v7-delta"
for s in development transfer; do
  $PY -m systemone.compare --candidate runs/e4b-v8/$s --reference runs/e4b-v7/$s --out runs/e4b-v8/$s/vs-v7.json > "$LOG/v8-compare-$s-v7.log" 2>&1 || stage "compare $s vs v7 failed"
  $PY -m systemone.compare --candidate runs/e4b-v8/$s --reference runs/e4b-v7-delta/$s --out runs/e4b-v8/$s/vs-delta.json > "$LOG/v8-compare-$s-delta.log" 2>&1 || stage "compare $s vs delta failed"
done
stage "extra: e4b-v7-delta transfer with date_facts"
$PY -m systemone.benchmark --run runs/e4b-v7-delta --suite evals/v4/transfer-v4 --date_facts --out runs/e4b-v7-delta/transfer-datefacts > "$LOG/delta-transfer-datefacts.log" 2>&1 || stage "delta date_facts benchmark failed"
stage "ALL DONE"
