#!/bin/bash
# Pilot run -> gate check -> full Kev-4B recipe -> evaluation -> calibration, unattended (nohup this script).
# Stage markers go to runs/logs/pipeline.status; each stage logs to runs/logs/<stage>.log.
#   pilot : decision-v7 training partition, 20% of the public records + all generated policy records, one epoch (~1 h on an M3 Max)
#   gate  : first 200 development records must score clean accuracy >= 0.50 AND agnews (4-way Choice) >= 0.45: the
#           first Gemma pilot scored 0.41 by learning only the yes/no questions, with every Choice source at chance
#   full  : the Kev-4B recipe (every training record, two epochs, LoRA r=16) with the Gemma settings: normalized head
#           inputs, head lr 1e-3, LoRA lr 1e-4 (~17 h: public records cost ~3.5 s each, generated ones ~0.5 s)
#   eval  : full development partition + transfer-v4 development (out of domain), then temperature calibration
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=runs/logs; mkdir -p "$LOG"
STATUS="$LOG/pipeline.status"
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
fail() { stage "FAILED: $*"; exit 1; }

stage "pilot: training runs/e4b-pilot"
$PY -m systemone.train --suite evals/v7/decision-v7 --public_frac 0.2 --epochs 1 --weights_dtype bf16 --lr 1e-4 --head_lr 1e-3 --head_norm 1 --readout delimiter --accum 8 --out runs/e4b-pilot > "$LOG/pilot-train.log" 2>&1 || fail "pilot training (see $LOG/pilot-train.log)"

stage "gate: scoring 200 development records"
$PY -m systemone.benchmark --run runs/e4b-pilot --suite evals/v7/decision-v7 --limit 200 --out runs/e4b-pilot/dev200 > "$LOG/pilot-gate.log" 2>&1 || fail "pilot benchmark (see $LOG/pilot-gate.log)"
GATE=$($PY -c "
import json; r = json.load(open('runs/e4b-pilot/dev200/report.json')); t = r['tasks']
acc, ag, bk = r['clean']['acc'], t.get('agnews', {}).get('acc', 0), t.get('banking77', {}).get('acc', 0)
print(f'clean={acc:.3f} agnews={ag:.3f} banking77={bk:.3f} pass={int(acc >= 0.50 and ag >= 0.45)}')")
stage "gate: pilot on 200 dev records: $GATE"
[[ "$GATE" == *"pass=1"* ]] || fail "gate: $GATE (need clean >= 0.50 and agnews >= 0.45), full run not started"

stage "full: training runs/e4b-v7 (Kev-4B recipe, 2 epochs)"
$PY -m systemone.train --suite evals/v7/decision-v7 --epochs 2 --weights_dtype bf16 --lr 1e-4 --head_lr 1e-3 --head_norm 1 --readout delimiter --accum 8 --out runs/e4b-v7 > "$LOG/full-train.log" 2>&1 || fail "full training (see $LOG/full-train.log)"

stage "eval: decision-v7 development (in distribution)"
$PY -m systemone.benchmark --run runs/e4b-v7 --suite evals/v7/decision-v7 --out runs/e4b-v7/development > "$LOG/full-dev.log" 2>&1 || fail "development benchmark"
stage "eval: transfer-v4 development (out of domain)"
$PY -m systemone.benchmark --run runs/e4b-v7 --suite evals/v4/transfer-v4 --out runs/e4b-v7/transfer > "$LOG/full-transfer.log" 2>&1 || fail "transfer benchmark"
stage "calibrate: fitting the temperature on development rows"
$PY scripts/calibrate_checkpoint.py --run runs/e4b-v7 --rows runs/e4b-v7/development/rows.json --transfer runs/e4b-v7/transfer/rows.json > "$LOG/full-calibrate.log" 2>&1 || fail "calibration"
stage "DONE: runs/e4b-v7 trained, evaluated and calibrated"
