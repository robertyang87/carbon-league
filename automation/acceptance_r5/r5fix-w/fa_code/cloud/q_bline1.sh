#!/bin/bash
# B line item 4 (plan_exit_loop.md sec. 6): SEQ_KEYS=2, 12 sims per car (24 per step), 3 training opponents x 100, seeds 994000
# -> paired with the B group (sx_S20_*). Pack fx_m2t3S12q2 = fx_m2t3S20n with FIX_SIMS (12,) + SEQ_KEYS 2.
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S12q2.py exp/$O.py 994000 100 sx_Q2s12_$O > ../data/sx_Q2s12_$O.out 2>&1 < /dev/null &
done
wait; echo BL1_DONE > ../data/bl1.done
