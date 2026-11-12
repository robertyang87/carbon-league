#!/bin/bash
# usage: dgx_ex1.sh TAG  (inputs ../data/<TAG>_in/*/ex1_*, dataset cnet2_ds_<TAG>, logs ../data/<TAG>_logs)
# DGX (spark-29db, ~/junyang only): search-distillation round ex1 -- build (steps <= 239, no awr_weights), lr0 baseline,
# mix A (hard 1, KL 30), mix B (hard 0.3, KL 100). Cores capped at 16 (80% of 20).
cd ~/junyang/carbon/code
PY=../.venv/bin/python; G=${1:-ex1}; L=../data/${G}_logs; mkdir -p $L
DIRS=$(ls -d ../data/${G}_in/*/ex1_* | grep -v '\.out$' | tr '\n' ' ')
echo "== build $(date +%T) $DIRS" > $L/run.log
CNET2_STEP_MAX=239 BUILD_WORKERS=16 $PY build_awr.py ../data/cnet2_ds_$G $DIRS > $L/build.log 2>&1 || { echo BUILD_FAIL >> $L/run.log; exit 1; }
tail -2 $L/build.log >> $L/run.log
T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=12 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_USE_W=0 BC_USE_UW=0"
run() { echo "== $1 $(date +%T)" >> $L/run.log; env $T $2 $PY cnet2_train.py ../data/cnet2_ds_$G ../data/weights/cnet2_b128${G}$1.npz $3 '{"C":128,"NB":8,"D":128}' > $L/train_$1.log 2>&1; grep '"ep"' $L/train_$1.log >> $L/run.log; }
run 0 "BC_SELF_KL=30 BC_EXIT_W=1 BC_LR=0" 1
echo BASELINE_DONE >> $L/run.log
$PY - $L/train_0.log <<'PY' >> $L/run.log || { echo BASELINE_OFF_STOP >> $L/run.log; exit 2; }
import json, sys
v = [json.loads(l)["val"] for l in open(sys.argv[1]) if l.startswith('{"ep"')][-1]
ok = 0.96 <= v["collector"] <= 0.99 and v["planter"] >= 0.97 and v["base"] >= 0.97
print("GATE", v, "ok" if ok else "OFF"); sys.exit(0 if ok else 1)
PY
run A "BC_SELF_KL=30 BC_EXIT_W=1 BC_LR=1e-4" 3
run B "BC_SELF_KL=100 BC_EXIT_W=0.3 BC_LR=1e-4" 3
echo ALL_DONE >> $L/run.log
