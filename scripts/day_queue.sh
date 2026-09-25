#!/bin/bash
# Today's GPU queue, one job at a time: wait for the inference-time evals, delta fine-tune + evals + paired comparison,
# smoke-test the multi-layer readout, then the pilot sweep. Results: runs/logs/queue.status, runs/sweep/results.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=runs/logs; STATUS=$LOG/queue.status
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
while pgrep -f infer_evals.sh >/dev/null; do sleep 30; done

# --- delta fine-tune of e4b-v7 on the targeted data ---------------------------------------------------------------
stage "delta: training runs/e4b-v7-delta from runs/e4b-v7 on evals/delta-v1 + 2000 replayed suite records"
rm -rf runs/e4b-v7-delta
$PY -m systemone.train --data evals/delta-v1/train.jsonl --suite evals/v7/decision-v7 --replay 2000 --init_from runs/e4b-v7 \
  --lr 2e-5 --head_lr 2e-5 --epochs 1 --weights_dtype bf16 --head_norm 1 --readout delimiter --accum 8 --save_every 200 --out runs/e4b-v7-delta > $LOG/delta-train.log 2>&1 \
  && stage "delta: trained" || stage "delta: FAILED training (see $LOG/delta-train.log)"
if [ -f runs/e4b-v7-delta/head.pt ]; then
  $PY -m systemone.benchmark --run runs/e4b-v7-delta --suite evals/v7/decision-v7 --out runs/e4b-v7-delta/development > $LOG/delta-dev.log 2>&1 || stage "delta: FAILED dev benchmark"
  $PY -m systemone.benchmark --run runs/e4b-v7-delta --suite evals/v4/transfer-v4 --out runs/e4b-v7-delta/transfer > $LOG/delta-transfer.log 2>&1 || stage "delta: FAILED transfer benchmark"
  $PY scripts/calibrate_checkpoint.py --run runs/e4b-v7-delta --rows runs/e4b-v7-delta/development/rows.json --transfer runs/e4b-v7-delta/transfer/rows.json > $LOG/delta-calibrate.log 2>&1 || stage "delta: FAILED calibration"
  for s in development transfer; do
    $PY -m systemone.compare --candidate runs/e4b-v7-delta/$s/rows.json --reference runs/e4b-v7/$s/rows.json --out runs/e4b-v7-delta/$s/vs-base.json > $LOG/delta-compare-$s.log 2>&1 || stage "delta: compare $s failed"
  done
  stage "delta: evaluated (runs/e4b-v7-delta/{development,transfer}/report.json, vs-base.json)"
fi

# --- multi-layer readout smoke test (fails fast before the sweep spends an hour on it) ------------------------------
rm -rf runs/smoke-layers
$PY -m systemone.train --suite evals/smoke-v1 --weights_dtype bf16 --accum 2 --epochs 1 --head_norm 1 --readout_layers 12,24 --out runs/smoke-layers > $LOG/smoke-layers.log 2>&1 \
  && $PY -m systemone.benchmark --run runs/smoke-layers --suite evals/smoke-v1 --limit 8 --out runs/smoke-layers/dev > $LOG/smoke-layers-bench.log 2>&1 \
  && stage "layers: smoke test passed" || stage "layers: FAILED smoke test (see $LOG/smoke-layers*.log)"

# --- pilot sweep: 1 epoch on 20% public + all generated records, gate on 400 dev records ----------------------------
mkdir -p runs/sweep; RES=runs/sweep/results.tsv
[ -f $RES ] || printf 'name\tclean\tagnews\tbanking77\tboolq\tece\tbrier\n' > $RES
COMMON="--suite evals/v7/decision-v7 --public_frac 0.2 --epochs 1 --weights_dtype bf16 --accum 8 --head_norm 1 --readout delimiter --save_every 200"
pilot() { name=$1; shift
  stage "sweep: $name ($*)"
  rm -rf "runs/sweep/$name"
  $PY -m systemone.train $COMMON "$@" --out "runs/sweep/$name" > "$LOG/sweep-$name.log" 2>&1 || { stage "sweep: $name FAILED training"; return; }
  $PY -m systemone.benchmark --run "runs/sweep/$name" --suite evals/v7/decision-v7 --limit 400 --out "runs/sweep/$name/dev400" > "$LOG/sweep-$name-bench.log" 2>&1 || { stage "sweep: $name FAILED benchmark"; return; }
  $PY - "$name" >> $RES <<'PY'
import json, sys; n = sys.argv[1]; r = json.load(open(f"runs/sweep/{n}/dev400/report.json")); c = r["clean"]; t = r["tasks"]
g = lambda s: f"{t[s]['acc']:.3f}" if s in t else "-"
print(f"{n}\t{c['acc']:.3f}\t{g('agnews')}\t{g('banking77')}\t{g('boolq')}\t{c['ece']:.3f}\t{c['brier']:.3f}")
PY
  stage "sweep: $name done: $(tail -1 $RES)"
}
pilot base            --lr 1e-4 --head_lr 2e-4 --head_warmup_steps 300
pilot layers12-24     --lr 1e-4 --head_lr 2e-4 --head_warmup_steps 300 --readout_layers 12,24
pilot rank32          --lr 1e-4 --head_lr 2e-4 --head_warmup_steps 300 --lora 32
pilot headlr5e-4-w200 --lr 1e-4 --head_lr 5e-4 --head_warmup_steps 200
pilot loralr5e-5      --lr 5e-5 --head_lr 2e-4 --head_warmup_steps 300
stage "QUEUE DONE"
