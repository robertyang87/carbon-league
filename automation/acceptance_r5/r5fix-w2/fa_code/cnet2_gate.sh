#!/bin/bash
# cnet2_gate.sh CAND.py TAG  -- coordinator's selection gate, 3 workers, appends to ../data/cnet2_gate_TAG.log
C=$1; T=$2; L=../data/cnet2_gate_$T.log
export EVO_WORKERS=2
../.venv/bin/python cnet2_eval.py $C exp/submission_cnet2_v12.py 973000 24 ${T}_vs_v12 >> $L 2>&1
for s in t0 t3 t2 t4; do
  ../.venv/bin/python cnet2_eval.py $C exp/cnet2_style_$s.py 973000 8 ${T}_vs_style_$s >> $L 2>&1
  ../.venv/bin/python cnet2_eval.py exp/submission_cnet2_v12.py exp/cnet2_style_$s.py 973000 8 v12_vs_style_$s >> $L 2>&1
done
../.venv/bin/python cnet2_eval.py $C exp/opp_1cz_v5.py 960000 32 ${T}_vs_v5 >> $L 2>&1
echo GATE_DONE >> $L
