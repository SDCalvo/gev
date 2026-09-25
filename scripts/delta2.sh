#!/bin/bash
# Second delta pass (evals/delta-v2: MCQ mix, paraphrase, 200 rule structures, dates-v2) on top of e4b-v8-delta, then evals and paired comparisons.
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=runs/logs; STATUS=$LOG/pipeline.status
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
stage "delta2: training runs/e4b-v8-delta2 from runs/e4b-v8"
rm -rf runs/e4b-v8-delta2
$PY -m systemone.train --data evals/delta-v2/train.jsonl --suite evals/v7/decision-v7 --replay 3000 --init_from runs/e4b-v8-delta --lr 2e-5 --head_lr 2e-5 --epochs 1 \
  --weights_dtype bf16 --head_norm 1 --readout delimiter --readout_layers 12,24 --brier_w 0.5 --accum 8 --save_every 200 --out runs/e4b-v8-delta2 > $LOG/delta2-train.log 2>&1 || { stage "delta2: FAILED training"; exit 1; }
for s in development:evals/v7/decision-v7 transfer:evals/v4/transfer-v4; do
  $PY -m systemone.benchmark --run runs/e4b-v8-delta2 --suite ${s#*:} --out runs/e4b-v8-delta2/${s%%:*} > $LOG/delta2-${s%%:*}.log 2>&1 || stage "delta2: FAILED ${s%%:*} benchmark"
done
$PY scripts/calibrate_checkpoint.py --run runs/e4b-v8-delta2 --rows runs/e4b-v8-delta2/development/rows.json --transfer runs/e4b-v8-delta2/transfer/rows.json > $LOG/delta2-calibrate.log 2>&1 || stage "delta2: FAILED calibration"
for s in development transfer; do for ref in e4b-v8-delta e4b-v8; do
  $PY -m systemone.compare --candidate runs/e4b-v8-delta2/$s --reference runs/$ref/$s --out runs/e4b-v8-delta2/$s/vs-${ref#e4b-}.json > /dev/null 2>&1 || stage "delta2: compare $s vs $ref failed"
done; done
stage "delta2: ALL DONE"
