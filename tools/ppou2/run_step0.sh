#!/bin/bash
# Step 0: frozen-start baseline fields (PREREG/start-order). Sequential, pinned env.
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
PY=/root/carbon/.venv/bin/python
L=/root/carbon/probe_m2/step0.log
echo "SMOKE_E2_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/m3s2r1_base.py exp/echo2.py 985000 4 ppou2_smoke_e2 >> $L 2>&1
echo "SMOKE_E2_RC=$? $(date '+%F %T %Z')" >> $L
echo "M2A_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 980000 2000 ppou2_base_m2_a >> $L 2>&1
echo "M2A_RC=$? $(date '+%F %T %Z')" >> $L
echo "M2B_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 980000 2000 ppou2_base_m2_b >> $L 2>&1
echo "M2B_RC=$? $(date '+%F %T %Z')" >> $L
echo "E2A_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/m3s2r1_base.py exp/echo2.py 982000 2000 ppou2_base_e2_a >> $L 2>&1
echo "E2A_RC=$? $(date '+%F %T %Z')" >> $L
echo DONE > /root/carbon/probe_m2/step0_done.txt
