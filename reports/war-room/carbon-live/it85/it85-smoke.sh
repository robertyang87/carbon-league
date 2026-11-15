#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/it85-smoke.log
echo "SMOKE START $(date '+%F %T %Z') cand" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/it85.py exp/opp_master2r.py 996000 2 it85_smoke 2>&1 | tee -a $LOG
echo "SMOKE cand DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE START $(date '+%F %T %Z') base" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1.py exp/opp_master2r.py 996000 2 it85base_smoke 2>&1 | tee -a $LOG
echo "SMOKE base DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
