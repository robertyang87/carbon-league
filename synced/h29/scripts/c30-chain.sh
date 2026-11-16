#!/bin/bash
# Chain launcher on spot-34: wait for it30 m-fields end marker, then run the
# ppou2_c_it30 makeup exam (C1, ordered 2026-11-15 22:54 by war room).
#  c30_m2: exp/ppou2_c_it30.py (c5fd2fd8) vs exp/opp_master2r.py (2aebfbc1), seed 980000, 1000
#  (split-half per owner share order 2026-11-15 23:43 + war room ruling 23:46: C1 runs the
#   FIRST half 980000-980999 of the Master2 makeup; C2 runs 981000-981999 on spot-35)
cd /root/carbon || exit 1
while [ ! -f /root/carbon/it30_end.txt ]; do sleep 20; done
sleep 5
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/c30_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_c_it30.py exp/opp_master2r.py 980000 1000 c30_m2
echo "RC_c30_m2=$?"
date +%s > /root/carbon/c30_end.txt
echo C30_DONE
