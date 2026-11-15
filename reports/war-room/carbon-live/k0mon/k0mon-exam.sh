#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/k0mon-exam.log
echo "START $(date '+%F %T %Z') field a0_vs_b0 K0-A0 vs K0-B0" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/k0_a0.py exp/k0_b0.py 990100 64 k0_a0_b0 2>&1 | tee -a $LOG
echo "FIELD a0_vs_b0 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field a1_vs_b0 K0-A1 vs K0-B0" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/k0_a1.py exp/k0_b0.py 990100 64 k0_a1_b0 2>&1 | tee -a $LOG
echo "FIELD a1_vs_b0 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
