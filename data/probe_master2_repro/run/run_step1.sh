#!/bin/bash
# C probe step 1: two replicate batches under the pinned config (PREREG v3).
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
PY=/root/carbon/.venv/bin/python
echo "P1_START $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step1.log
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 probe_m2_p1 >> /root/carbon/probe_m2/step1.log 2>&1
echo "P1_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step1.log
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 probe_m2_p2 >> /root/carbon/probe_m2/step1.log 2>&1
echo "P2_RC=$? $(date '+%F %T %Z')" >> /root/carbon/probe_m2/step1.log
echo DONE > /root/carbon/probe_m2/step1_done.txt
