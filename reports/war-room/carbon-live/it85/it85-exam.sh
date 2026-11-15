#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/it85-exam.log
echo "START $(date '+%F %T %Z') field cand_m2 it85 vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/it85.py exp/opp_master2r.py 996000 200 it85_m2 2>&1 | tee -a $LOG
echo "FIELD cand_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field base_m2 m3s2r1 vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1.py exp/opp_master2r.py 996000 200 it85base_m2 2>&1 | tee -a $LOG
echo "FIELD base_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
