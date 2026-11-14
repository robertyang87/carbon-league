#!/bin/bash
# C probe step 3 (single-variable harness swap to dceb1ebe), chained after amendment batches.
while [ ! -f /root/carbon/probe_m2/step2b_done.txt ]; do sleep 20; done
sleep 5
cd /root/carbon/probe_m2/code_oldh || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
echo "P4_START $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step3.log
/root/carbon/.venv/bin/python bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 probe_m2_p4 >> /root/carbon/probe_m2/step3.log 2>&1
echo "P4_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step3.log
echo DONE > /root/carbon/probe_m2/step3_done.txt
