#!/bin/bash
# C probe amendment batches (exam-day true EVO_WORKERS=62), chained after step 2 (P3).
while [ ! -f /root/carbon/probe_m2/step2_done.txt ]; do sleep 20; done
sleep 5
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=62
echo "P1B_START $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step2b.log
/root/carbon/.venv/bin/python bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 probe_m2_p1b >> /root/carbon/probe_m2/step2b.log 2>&1
echo "P1B_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step2b.log
/root/carbon/.venv/bin/python bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 probe_m2_p2b >> /root/carbon/probe_m2/step2b.log 2>&1
echo "P2B_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step2b.log
echo DONE > /root/carbon/probe_m2/step2b_done.txt
