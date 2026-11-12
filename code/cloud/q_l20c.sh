#!/bin/bash
# L20-3 (2026-11-11 03:xx): round 1 again with the soft self-anchor (BC_SELF_KL, 45e3a83) on the round-1 dataset,
# same settings as the first L20 run otherwise; then strength 100 if the collector drop is 0.5-1.7 pt; then build rounds 1+2.
cd /root/carbon/code; L=/root/carbon/data/train_k30.log; P=/root/carbon/.venv/bin/python
{ echo "== torch install $(date +%T)"; $P -m pip uninstall -y -q torch 2>/dev/null; $P -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
  echo "== torch $($P -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))')"
  until [ -f /root/carbon/data/DATA_READY ]; do sleep 15; done
  D=../data/cnet2_ds_sp1; SRC="../data/sp1_m2s ../data/sp1_m2so ../data/sp1_scout1"; rm -rf $D
  echo "== build $(date +%T)"; CNET2_STEP_MAX=300 BUILD_WORKERS=62 $P build_awr.py $D $SRC 2>&1 | tail -3
  echo "== weights $(date +%T)"; $P awr_weights.py $D $SRC 2>&1 | tail -4
  T="BC_DEVICE=cuda BC_RAM=1 BC_THREADS=16 BC_SAVE_EPOCHS=0 BC_INIT=../data/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1"
  for k in 30 100; do
    echo "== train self_kl $k lr 1e-4 x2 $(date +%T)"; env $T BC_SELF_KL=$k BC_LR=1e-4 $P cnet2_train.py $D ../data/weights/cnet2_b128sp1k$k.npz 2 '{"C":128,"NB":8,"D":128}' 2>&1 | grep -v '"i":' | tail -10
    c=$(grep '"ep": 1' $L | tail -1 | sed 's/.*"collector": \([0-9.]*\).*/\1/')
    echo "== k$k collector $c $(date +%T)"; touch ../data/K${k}_DONE
    awk -v c="$c" 'BEGIN{exit !(c >= 0.975 && c < 0.9869)}' || break
  done
  D2=../data/cnet2_ds_sp12; SRC2="$SRC ../data/sp2_m2s ../data/sp2_m2so ../data/sp2_scout1"; rm -rf $D2
  echo "== build rounds 1+2 $(date +%T)"; CNET2_STEP_MAX=300 BUILD_WORKERS=62 $P build_awr.py $D2 $SRC2 2>&1 | tail -3
  echo "== weights $(date +%T)"; $P awr_weights.py $D2 $SRC2 2>&1 | tail -4
  date; echo ALL_DONE; } > $L 2>&1
