#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/a4r1-exam.log
echo "START $(date '+%F %T %Z') field a4_vs_m2 m3s2r1_a4(ep18) vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_a4.py exp/opp_master2r.py 996000 200 a4r1_a4_m2 2>&1 | tee -a $LOG
echo "FIELD a4_vs_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field base_vs_m2 m3s2r1_base(ep19) vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 a4r1_base_m2 2>&1 | tee -a $LOG
echo "FIELD base_vs_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
