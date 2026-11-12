#!/bin/bash
# narrow camper-bust (CB_STEP 270, CB_ER 2, CB_R 4, CB_MAXC 5, CB_LOAD 100, CB_D 4) vs control, both packs from d043953 source
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for A in S20m S20cbn; do for O in opp_1cz_v9 cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=8 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3$A.py exp/$O.py 994000 100 cbn_${A}_$O > ../data/cbn_${A}_$O.out 2>&1 < /dev/null &
done; done
wait; echo CBN_DONE > ../data/cbn.done
