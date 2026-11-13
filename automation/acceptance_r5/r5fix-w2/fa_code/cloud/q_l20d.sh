#!/bin/bash
# L20-3 (2026-11-11 04:xx): rounds 1+2+3 (22500 whole games) with the soft self-anchor. Starts after q_l20c.sh is done and the
# round-3 corpora are uploaded (DATA3_READY). Checks game keys are unique, measures a fresh lr-0 baseline on this dataset,
# trains strength 30 (100 if the collector drops > 0.5 pt vs that baseline), then the pure-net exam (new arm only, 994000 x 200).
cd /root/carbon/code; L=/root/carbon/data/train_sp123.log; P=/root/carbon/.venv/bin/python
until grep -q ALL_DONE ../data/train_k30.log 2>/dev/null && [ -f ../data/DATA3_READY ] && [ -f ../data/EXAM_K30_DONE ]; do sleep 20; done
{ SRC=""; for r in sp1 sp2 sp3; do for o in m2s m2so scout1; do SRC="$SRC ../data/${r}_$o"; done; done
  echo "== keys: total $(ls $(for d in $SRC; do echo $d/replays; done) | grep -c json.gz) distinct $(ls $(for d in $SRC; do echo $d/replays; done) | grep json.gz | sort -u | wc -l)"
  D=../data/cnet2_ds_sp123; rm -rf $D ../data/cnet2_ds_sp12
  echo "== build $(date +%T)"; CNET2_STEP_MAX=300 BUILD_WORKERS=62 $P build_awr.py $D $SRC 2>&1 | tail -3
  echo "== weights $(date +%T)"; $P awr_weights.py $D $SRC 2>&1 | tail -4
  T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=16 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
  echo "== baseline lr 0 $(date +%T)"; env $T BC_LR=0 $P cnet2_train.py $D ../data/weights/cnet2_b128sp123_lr0.npz 1 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -3
  b=$(grep '"ep": 0' $L | head -1 | sed 's/.*"collector": \([0-9.]*\).*/\1/')
  for k in 30 100; do
    echo "== train self_kl $k lr 1e-4 x2 $(date +%T)"; env $T BC_SELF_KL=$k BC_LR=1e-4 $P cnet2_train.py $D ../data/weights/cnet2_b128sp123k$k.npz 2 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -6
    c=$(grep '"ep": 1' $L | tail -1 | sed 's/.*"collector": \([0-9.]*\).*/\1/'); K=$k
    echo "== k$k collector $c baseline $b $(date +%T)"
    awk -v c="$c" -v b="$b" 'BEGIN{exit !(b - c > 0.005)}' || break
  done
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
  echo "== exam k$K $(date +%T)"
  for O in cnet2_m2s2stand cnet2_m2so s1_net cnet2_m2sstand; do
    AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128sp123k$K.npz EVO_WORKERS=15 ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 sp123x_k${K}_$O > /dev/null 2>&1 < /dev/null &
  done; wait
  date; echo ALL_DONE; } > $L 2>&1
