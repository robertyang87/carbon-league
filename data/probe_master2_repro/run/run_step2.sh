#!/bin/bash
# C probe step 2 (single-variable package swap), chained after Step 0 completes.
while [ ! -f /root/carbon/probe_m2/step0_done.txt ]; do sleep 30; done
sleep 5
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
echo "P3_START $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step2.log
/root/carbon/.venv/bin/python bigeval.py exp/m3s2r1.py exp/opp_master2r.py 996000 200 probe_m2_p3 >> /root/carbon/probe_m2/step2.log 2>&1
echo "P3_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step2.log
echo DONE > /root/carbon/probe_m2/step2_done.txt
