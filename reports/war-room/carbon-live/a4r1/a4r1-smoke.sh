#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/a4r1-smoke.log
echo "SMOKE START $(date '+%F %T %Z') a4_vs_m2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_a4.py exp/opp_master2r.py 996000 2 a4r1_a4_smoke 2>&1 | tee -a $LOG
echo "SMOKE a4 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE START $(date '+%F %T %Z') base_vs_m2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 2 a4r1_base_smoke 2>&1 | tee -a $LOG
echo "SMOKE base DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
