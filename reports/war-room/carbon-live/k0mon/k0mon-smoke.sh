#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/k0mon-smoke.log
echo "SMOKE START $(date '+%F %T %Z') a0_vs_b0" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/k0_a0.py exp/k0_b0.py 990100 2 k0_a0_smoke 2>&1 | tee -a $LOG
echo "SMOKE a0 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE START $(date '+%F %T %Z') a1_vs_b0" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/k0_a1.py exp/k0_b0.py 990100 2 k0_a1_smoke 2>&1 | tee -a $LOG
echo "SMOKE a1 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
