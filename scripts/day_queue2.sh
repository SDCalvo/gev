#!/bin/bash
# Second sweep batch, chained after scripts/day_queue.sh: loss shaping and the instruction-tuned base. Same protocol
# (1 epoch on 20% public + all generated records, gate on 400 dev records), results appended to runs/sweep/results.tsv.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; LOG=runs/logs; STATUS=$LOG/queue.status; RES=runs/sweep/results.tsv
stage() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
while pgrep -f "scripts/day_queue.sh" >/dev/null; do sleep 60; done
COMMON="--suite evals/v7/decision-v7 --public_frac 0.2 --epochs 1 --weights_dtype bf16 --accum 8 --head_norm 1 --readout delimiter --save_every 200 --lr 1e-4 --head_lr 2e-4 --head_warmup_steps 300"
pilot() { name=$1; shift
  stage "sweep2: $name ($*)"
  rm -rf "runs/sweep/$name"
  $PY -m systemone.train $COMMON "$@" --out "runs/sweep/$name" > "$LOG/sweep-$name.log" 2>&1 || { stage "sweep2: $name FAILED training"; return; }
  $PY -m systemone.benchmark --run "runs/sweep/$name" --suite evals/v7/decision-v7 --limit 400 --out "runs/sweep/$name/dev400" > "$LOG/sweep-$name-bench.log" 2>&1 || { stage "sweep2: $name FAILED benchmark"; return; }
  $PY - "$name" >> $RES <<'PY'
import json, sys; n = sys.argv[1]; r = json.load(open(f"runs/sweep/{n}/dev400/report.json")); c = r["clean"]; t = r["tasks"]
g = lambda s: f"{t[s]['acc']:.3f}" if s in t else "-"
print(f"{n}\t{c['acc']:.3f}\t{g('agnews')}\t{g('banking77')}\t{g('boolq')}\t{c['ece']:.3f}\t{c['brier']:.3f}")
PY
  stage "sweep2: $name done: $(tail -1 $RES)"
}
pilot brierw0.5   --brier_w 0.5
pilot permkl0.5   --perm_kl 0.5 --perm_frac 0.3
pilot ordw0.5     --ord_w 0.5
IT_SHA=$(cat runs/logs/gemma-it.sha 2>/dev/null || true)
if [ -n "$IT_SHA" ] && [ -d "$HOME/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots" ]; then
  pilot itbase      --base google/gemma-4-E4B-it --base_revision "$IT_SHA"
else
  stage "sweep2: itbase skipped (instruction-tuned checkpoint not downloaded)"
fi
stage "QUEUE2 DONE"
