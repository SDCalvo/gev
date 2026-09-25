#!/bin/bash
# Delta pass on top of e4b-v8 (same targeted data as the v7 delta), then evals and paired comparisons.
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=runs/logs; STATUS=$LOG/pipeline.status
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
stage "delta-v8: training runs/e4b-v8-delta from runs/e4b-v8"
rm -rf runs/e4b-v8-delta
$PY -m systemone.train --data evals/delta-v1/train.jsonl --suite evals/v7/decision-v7 --replay 2000 --init_from runs/e4b-v8 --lr 2e-5 --head_lr 2e-5 --epochs 1 \
  --weights_dtype bf16 --head_norm 1 --readout delimiter --readout_layers 12,24 --brier_w 0.5 --accum 8 --save_every 200 --out runs/e4b-v8-delta > $LOG/delta-v8-train.log 2>&1 || { stage "delta-v8: FAILED training"; exit 1; }
for s in development:evals/v7/decision-v7 transfer:evals/v4/transfer-v4; do
  $PY -m systemone.benchmark --run runs/e4b-v8-delta --suite ${s#*:} --out runs/e4b-v8-delta/${s%%:*} > $LOG/delta-v8-${s%%:*}.log 2>&1 || stage "delta-v8: FAILED ${s%%:*} benchmark"
done
$PY scripts/calibrate_checkpoint.py --run runs/e4b-v8-delta --rows runs/e4b-v8-delta/development/rows.json --transfer runs/e4b-v8-delta/transfer/rows.json > $LOG/delta-v8-calibrate.log 2>&1 || stage "delta-v8: FAILED calibration"
for s in development transfer; do for ref in e4b-v8 e4b-v7-delta; do
  $PY -m systemone.compare --candidate runs/e4b-v8-delta/$s --reference runs/$ref/$s --out runs/e4b-v8-delta/$s/vs-${ref#e4b-}.json > /dev/null 2>&1 || stage "delta-v8: compare $s vs $ref failed"
done; done
stage "delta-v8: ALL DONE"
