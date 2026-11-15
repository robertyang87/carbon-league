#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/waux1-exam.log
echo "START $(date '+%F %T %Z') field m2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m9xj570waux1.py exp/opp_master2r.py 996000 200 m9xj570waux1_m2 2>&1 | tee -a $LOG
echo "FIELD m2 DONE rc=$? $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field echo2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m9xj570waux1.py exp/echo2.py 999000 200 m9xj570waux1_echo2 2>&1 | tee -a $LOG
echo "FIELD echo2 DONE rc=$? $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
