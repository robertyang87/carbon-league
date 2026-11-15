#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/it95-smoke.log
echo "SMOKE START $(date '+%F %T %Z') cand_vs_m2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/it95.py exp/opp_master2r.py 996000 2 it95_smoke 2>&1 | tee -a $LOG
echo "SMOKE cand DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE START $(date '+%F %T %Z') base_vs_m2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 2 it95base_smoke 2>&1 | tee -a $LOG
echo "SMOKE base DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "SMOKE ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
