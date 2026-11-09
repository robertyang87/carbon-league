#!/bin/bash
# AWR round over several corpora: host-4 corpora (H4 list) + Mac corpora (MAC list) + DGX corpora (DGX list, already
# there) -> DGX; build, AWR weights (all corpora pooled), + anchor, decoder-only training from INIT; opening exam on Mac.
#   usage: awr_round_multi.sh TAG INIT_WEIGHTS "h4 corpora" "mac corpora" "dgx corpora" [COPY_TEACH]
T=$1; INIT=$2; H4C=$3; MACC=$4; DGXC=$5; CP=${6:-5:2}; [ "$CP" = none ] && CP=""      # none: keep the init's style rows
K=$HOME/.ssh/carbon_aliyun_ed25519; H=root@47.117.170.99; L=/Users/baggioustb/dev/carbon/carbon-league/data/awr_round_$T.log
cd /Users/baggioustb/dev/carbon/carbon-league/data
for c in $H4C; do until ssh -i $K -o ConnectTimeout=40 $H "test -f ~/carbon/data/$c/index.json" </dev/null; do sleep 30; done; done
for c in $MACC; do until [ -f $c/index.json ]; do sleep 30; done; done
echo "$(date +%T) all generated" >> $L
for c in $H4C; do ssh -i $K $H "cd ~/carbon/data; tar cf $c.tar $c" </dev/null && scp -q -i $K ${H}:/root/carbon/data/$c.tar /tmp/ && scp -q /tmp/$c.tar dgx-via-spot3:junyang/carbon/data/ && rm -f /tmp/$c.tar && ssh dgx-via-spot3 "cd ~/junyang/carbon/data; tar xf $c.tar && rm $c.tar" </dev/null && echo "$(date +%T) $c on dgx" >> $L; done
for c in $MACC; do tar cf /tmp/$c.tar $c && scp -q /tmp/$c.tar dgx-via-spot3:junyang/carbon/data/ && rm -f /tmp/$c.tar && ssh dgx-via-spot3 "cd ~/junyang/carbon/data; tar xf $c.tar && rm $c.tar" </dev/null && echo "$(date +%T) $c on dgx" >> $L; done
ALL=""; for c in $H4C $MACC $DGXC; do ALL="$ALL ../data/$c"; done
EXTRA=""; [ -f r2_extra.flag ] && EXTRA=cnet2_ds_m2vsM && echo "$(date +%T) + Master2-vs-M opening rows (w 5)" >> $L
ssh -o ConnectTimeout=60 dgx-via-spot3 "cd ~/junyang/carbon/code; rm -rf ../data/cnet2_ds_awr_$T ../data/cnet2_ds_awrA_$T;
  CNET2_STEP_MAX=100 BUILD_WORKERS=14 ../.venv/bin/python build_awr.py ../data/cnet2_ds_awr_$T $ALL 2>&1 | tail -1 | cut -c1-200;
  ../.venv/bin/python awr_weights.py ../data/cnet2_ds_awr_$T $ALL;
  ../.venv/bin/python concat_ds.py ../data/cnet2_ds_anchor170k ../data/cnet2_ds_awr_$T ../data/cnet2_ds_awrA_$T 1 | tail -1;
  if [ -n \"$EXTRA\" ]; then ../.venv/bin/python concat_ds.py ../data/cnet2_ds_awrA_$T ../data/$EXTRA ../data/cnet2_ds_awrAX_$T 1 | tail -1; rm -rf ../data/cnet2_ds_awrA_$T; mv ../data/cnet2_ds_awrAX_$T ../data/cnet2_ds_awrA_$T; fi;
  BC_DEVICE=cuda BC_RAM=1 BC_THREADS=8 BC_SAVE_EPOCHS=0 BC_INIT=$INIT ${CP:+BC_COPY_TEACH=$CP} BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_LR=1e-4 \
    ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_awrA_$T ../data/weights/cnet2_b128awr_$T.npz 2 '{\"C\":128,\"NB\":8,\"D\":128}' 2>&1 | grep -E 'val|best|Error|Trace' | tail -3" </dev/null >> $L 2>&1
echo "$(date +%T) trained" >> $L
rsync -az dgx-via-spot3:junyang/carbon/data/weights/cnet2_b128awr_$T.npz weights/ || { echo NO_WEIGHTS >> $L; exit 1; }
cd ../code; python3 - "$T" <<'PY'
import sys
T = sys.argv[1]; s = open("exp/cnet2_dg5e5s57_240.py").read()
s = s.replace("'cnet2_b128dg5_ep5.npz'", f"'cnet2_b128awr_{T}.npz'").replace("'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}", "'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}")
assert f"awr_{T}" in s; open(f"exp/cnet2_awr{T}.py", "w").write(s)
PY
cd ../data
[ -f bigeval_live/o100_m2s2_dg5e5s57.jsonl ] && grep -q pd_clash bigeval_live/o100_m2s2_dg5e5s57.jsonl || bash open100.sh dg5e5s57 m2s2 996000 400
bash open100.sh awr$T m2s2 996000 400
[ -f bigeval_live/o100_m2r_dg5e5s57.jsonl ] && grep -q pd_clash bigeval_live/o100_m2r_dg5e5s57.jsonl || EVAL_OPP_CASH=300 bash open100.sh dg5e5s57 m2r 996000 400
EVAL_OPP_CASH=300 bash open100.sh awr$T m2r 996000 400
python3 open100_table.py m2s2 dg5e5s57 awr$T >> $L 2>&1; python3 open100_table.py m2r dg5e5s57 awr$T >> $L 2>&1
echo "$(date +%T) ROUND_DONE" >> $L
