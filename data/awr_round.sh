#!/bin/bash
# AWR round: CORPUS (on host 4) -> Mac -> DGX; build slot-2 dataset, AWR weights, + anchor, decoder-only training;
# then the opening exam on the Mac vs both stand-ins.  usage: awr_round.sh CORPUS TAG
C=$1; T=$2; K=$HOME/.ssh/carbon_aliyun_ed25519; H=root@47.117.170.99; L=/Users/baggioustb/dev/carbon/carbon-league/data/awr_round_$T.log
cd /Users/baggioustb/dev/carbon/carbon-league/data
until ssh -i $K -o ConnectTimeout=40 $H "test -f ~/carbon/data/$C/index.json" </dev/null; do sleep 30; done
echo "$(date +%T) generated" >> $L; ssh -i $K $H "tail -1 ~/carbon/data/$C.log" </dev/null >> $L
ssh -i $K $H "cd ~/carbon/data; tar cf $C.tar $C" </dev/null && scp -q -i $K ${H}:/root/carbon/data/$C.tar /tmp/ && echo "$(date +%T) on mac $(du -h /tmp/$C.tar | cut -f1)" >> $L
scp -q /tmp/$C.tar dgx-via-spot3:junyang/carbon/data/ && rm -f /tmp/$C.tar && echo "$(date +%T) on dgx" >> $L
ssh -o ConnectTimeout=60 dgx-via-spot3 "cd ~/junyang/carbon/data; tar xf $C.tar && rm $C.tar; cd ../code; rm -rf ../data/cnet2_ds_awr_$T ../data/cnet2_ds_awrA_$T;
  CNET2_STEP_MAX=100 BUILD_WORKERS=14 ../.venv/bin/python build_awr.py ../data/cnet2_ds_awr_$T ../data/$C 2>&1 | tail -1;
  ../.venv/bin/python awr_weights.py ../data/cnet2_ds_awr_$T ../data/$C;
  ../.venv/bin/python concat_ds.py ../data/cnet2_ds_anchor170k ../data/cnet2_ds_awr_$T ../data/cnet2_ds_awrA_$T 1 | tail -1;
  BC_DEVICE=cuda BC_RAM=1 BC_THREADS=8 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128dg5_ep5.npz BC_COPY_TEACH=5:2 BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_LR=1e-4 \
    ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_awrA_$T ../data/weights/cnet2_b128awr_$T.npz 2 '{\"C\":128,\"NB\":8,\"D\":128}' 2>&1 | grep -E 'val|best|Error|Trace' | tail -3" </dev/null >> $L 2>&1
echo "$(date +%T) trained" >> $L
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128awr_$T.npz weights/ || { echo NO_WEIGHTS >> $L; exit 1; }
cd ../code; python3 - "$T" <<'PY'
import sys
T = sys.argv[1]; s = open("exp/cnet2_dg5e5s57_240.py").read()
s = s.replace("'cnet2_b128dg5_ep5.npz'", f"'cnet2_b128awr_{T}.npz'").replace("'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}", "'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}")
assert f"awr_{T}" in s; open(f"exp/cnet2_awr{T}.py", "w").write(s)
PY
cd ../data; export EVO_WORKERS=4
bash open100.sh dg5e5s57 m2s2 996000 400; bash open100.sh awr$T m2s2 996000 400
EVAL_OPP_CASH=300 bash open100.sh dg5e5s57 m2r 996000 400; EVAL_OPP_CASH=300 bash open100.sh awr$T m2r 996000 400
python3 open100_table.py m2s2 dg5e5s57 awr$T >> $L 2>&1; python3 open100_table.py m2r dg5e5s57 awr$T >> $L 2>&1
echo "$(date +%T) ROUND_DONE" >> $L
