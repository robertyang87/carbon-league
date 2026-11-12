#!/bin/bash
# finish a round whose dataset cnet2_ds_awrA_TAG is already built on the DGX: cached decoder-only training, weights back, opening exam
T=$1; INIT=${2:-../data/weights/cnet2_b128dg5_ep5.npz}; CP=${3:-5:2}; L=/Users/baggioustb/dev/carbon/carbon-league/data/awr_round_$T.log
cd /Users/baggioustb/dev/carbon/carbon-league/data
echo "$(date +%T) cached training" >> $L
ssh -o ConnectTimeout=60 dgx-via-spot3 "cd ~/junyang/carbon/code; BC_DEVICE=cuda BC_RAM=1 BC_THREADS=8 BC_INIT=$INIT ${CP:+BC_COPY_TEACH=$CP} BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_LR=1e-4 \
  ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_awrA_$T ../data/weights/cnet2_b128awr_$T.npz 2 '{\"C\":128,\"NB\":8,\"D\":128}' 2>&1 | grep -E 'trunk_cache|val|best|Error|Trace' | tail -4" </dev/null >> $L 2>&1
echo "$(date +%T) trained" >> $L
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128awr_$T.npz weights/ || { echo NO_WEIGHTS >> $L; exit 1; }
cd ../code; python3 - "$T" <<'PY'
import sys
T = sys.argv[1]; s = open("exp/cnet2_dg5e5s57_240.py").read()
s = s.replace("'cnet2_b128dg5_ep5.npz'", f"'cnet2_b128awr_{T}.npz'").replace("'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}", "'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}")
open(f"exp/cnet2_awr{T}.py", "w").write(s)
PY
cd ../data; bash open100.sh awr$T m2s2 996000 400; EVAL_OPP_CASH=300 bash open100.sh awr$T m2r 996000 400
python3 open100_table.py m2s2 dg5e5s57 awrr1 awr$T >> $L 2>&1; python3 open100_table.py m2r dg5e5s57 awrr1 awr$T >> $L 2>&1
echo "$(date +%T) ROUND_DONE" >> $L
