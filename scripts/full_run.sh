#!/bin/bash
# The stages after a passed pilot gate: full training (resumable), evaluation, calibration. Unattended (nohup this).
# Training checkpoints every 100 optimizer steps into runs/e4b-v7/ckpt; a crash resumes from the last checkpoint, up
# to MAX_RETRIES times. Stage markers go to runs/logs/pipeline.status; each stage logs to runs/logs/full-*.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/logs; mkdir -p "$LOG"
STATUS="$LOG/pipeline.status"
MAX_RETRIES=${MAX_RETRIES:-6}
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
fail() { stage "FAILED: $*"; exit 1; }
TRAIN="$PY -m systemone.train --suite evals/v7/decision-v7 --epochs 2 --weights_dtype bf16 --lr 1e-4 --head_lr 2e-4 --head_norm 1 --head_warmup_steps 400 --readout delimiter --accum 8 --save_every 100 --out runs/e4b-v7"

attempt=0
if [ -f runs/e4b-v7/head.pt ]; then
  stage "full: runs/e4b-v7 already trained, skipping to evaluation"
else
  if [ -d runs/e4b-v7/ckpt ]; then stage "full: resuming runs/e4b-v7 from its checkpoint"; $TRAIN --resume >> "$LOG/full-train.log" 2>&1
  else stage "full: training runs/e4b-v7 (Kev-4B recipe, 2 epochs, checkpoint every 100 steps)"; $TRAIN > "$LOG/full-train.log" 2>&1; fi
  while [ ! -f runs/e4b-v7/head.pt ] && [ $attempt -lt $MAX_RETRIES ]; do
    attempt=$((attempt + 1))
    stage "full: training exited without a final save (attempt $attempt of $MAX_RETRIES), resuming from the checkpoint"
    tail -3 "$LOG/full-train.log" >> "$STATUS"
    [ -d runs/e4b-v7/ckpt ] || fail "no checkpoint to resume from"
    sleep 20
    $TRAIN --resume >> "$LOG/full-train.log" 2>&1
  done
  [ -f runs/e4b-v7/head.pt ] || fail "full training did not finish after $MAX_RETRIES resumes (see $LOG/full-train.log)"
fi

stage "eval: decision-v7 development (in distribution)"
$PY -m systemone.benchmark --run runs/e4b-v7 --suite evals/v7/decision-v7 --out runs/e4b-v7/development > "$LOG/full-dev.log" 2>&1 || fail "development benchmark"
stage "eval: transfer-v4 development (out of domain)"
$PY -m systemone.benchmark --run runs/e4b-v7 --suite evals/v4/transfer-v4 --out runs/e4b-v7/transfer > "$LOG/full-transfer.log" 2>&1 || fail "transfer benchmark"
stage "calibrate: fitting the temperature on development rows"
$PY scripts/calibrate_checkpoint.py --run runs/e4b-v7 --rows runs/e4b-v7/development/rows.json --transfer runs/e4b-v7/transfer/rows.json > "$LOG/full-calibrate.log" 2>&1 || fail "calibration"
stage "DONE: runs/e4b-v7 trained, evaluated and calibrated"
