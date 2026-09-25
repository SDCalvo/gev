#!/bin/bash
# After delta2: pilot the instruction-tuned base combined with the multi-layer readout (the two best sweep pilots, never
# tried together). If it beats the layers12-24 pilot (0.734 on the 400-record gate), run the full v8 recipe on the
# instruction-tuned base overnight (runs/e4b-v9), then the two delta passes on top, evals and comparisons.
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; LOG=runs/logs; STATUS=$LOG/pipeline.status; RES=runs/sweep/results.tsv
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
while pgrep -f "scripts/delta2.sh" >/dev/null; do sleep 60; done
IT="--base google/gemma-4-E4B-it --base_revision $(cat runs/logs/gemma-it.sha)"
FLAGS="--weights_dtype bf16 --head_norm 1 --readout delimiter --readout_layers 12,24 --brier_w 0.5 --accum 8 --lr 1e-4 --head_lr 2e-4"
stage "pilot: it+layers12-24"
rm -rf runs/sweep/it-layers
$PY -m systemone.train $IT $FLAGS --suite evals/v7/decision-v7 --public_frac 0.2 --epochs 1 --head_warmup_steps 300 --save_every 200 --out runs/sweep/it-layers > $LOG/sweep-it-layers.log 2>&1 || { stage "pilot: it+layers FAILED"; exit 1; }
$PY -m systemone.benchmark --run runs/sweep/it-layers --suite evals/v7/decision-v7 --limit 400 --out runs/sweep/it-layers/dev400 > $LOG/sweep-it-layers-bench.log 2>&1 || { stage "pilot: it+layers benchmark FAILED"; exit 1; }
ACC=$($PY -c "import json; r=json.load(open('runs/sweep/it-layers/dev400/report.json')); c=r['clean']; t=r['tasks']; print(f\"it-layers\t{c['acc']:.3f}\t{t['agnews']['acc']:.3f}\t{t['banking77']['acc']:.3f}\t{t['boolq']['acc']:.3f}\t{c['ece']:.3f}\t{c['brier']:.3f}\")")
echo "$ACC" >> $RES; stage "pilot: it+layers done: $ACC"
$PY -c "import sys; sys.exit(0 if float('$ACC'.split()[1]) > 0.734 else 1)" || { stage "pilot: it+layers does not beat 0.734, no overnight run"; exit 0; }

stage "v9: full v8 recipe on the instruction-tuned base -> runs/e4b-v9"
rm -rf runs/e4b-v9
$PY -m systemone.train $IT $FLAGS --data evals/v8-data/train.jsonl --suite evals/v7/decision-v7 --replay 12576 --epochs 1 --head_warmup_steps 400 --save_every 100 --out runs/e4b-v9 > $LOG/v9-train.log 2>&1
for i in 1 2 3; do [ -f runs/e4b-v9/head.pt ] && break; stage "v9: resuming (attempt $i)"; sleep 20; $PY -m systemone.train $IT $FLAGS --data evals/v8-data/train.jsonl --suite evals/v7/decision-v7 --replay 12576 --epochs 1 --head_warmup_steps 400 --save_every 100 --out runs/e4b-v9 --resume >> $LOG/v9-train.log 2>&1; done
[ -f runs/e4b-v9/head.pt ] || { stage "v9: FAILED training"; exit 1; }
prev=runs/e4b-v9
for d in delta-v1:2000 delta-v2:3000; do
  name=runs/e4b-v9-${d%%:*}
  stage "v9: delta pass ${d%%:*} -> $name"
  rm -rf $name
  $PY -m systemone.train $IT $FLAGS --data evals/${d%%:*}/train.jsonl --suite evals/v7/decision-v7 --replay ${d#*:} --init_from $prev --lr 2e-5 --head_lr 2e-5 --epochs 1 --save_every 200 --out $name > $LOG/v9-${d%%:*}-train.log 2>&1 || { stage "v9: delta ${d%%:*} FAILED"; exit 1; }
  prev=$name
done
for run in runs/e4b-v9 $prev; do
  for s in development:evals/v7/decision-v7 transfer:evals/v4/transfer-v4; do
    $PY -m systemone.benchmark --run $run --suite ${s#*:} --out $run/${s%%:*} > $LOG/$(basename $run)-${s%%:*}.log 2>&1 || stage "$run: ${s%%:*} benchmark FAILED"
  done
  $PY scripts/calibrate_checkpoint.py --run $run --rows $run/development/rows.json --transfer $run/transfer/rows.json > $LOG/$(basename $run)-calibrate.log 2>&1 || stage "$run: calibration FAILED"
  for s in development transfer; do $PY -m systemone.compare --candidate $run/$s --reference runs/e4b-v8-delta/$s --out $run/$s/vs-v8-delta.json > /dev/null 2>&1 || true; done
  stage "$run: evaluated"
done
stage "EVENING QUEUE DONE"
