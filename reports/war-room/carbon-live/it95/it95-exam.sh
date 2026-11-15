#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/it95-exam.log
echo "START $(date '+%F %T %Z') field cand_vs_m2 it95(snipe_n1 it95 frozen) vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/it95.py exp/opp_master2r.py 996000 200 it95_m2 2>&1 | tee -a $LOG
echo "FIELD cand_vs_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field base_vs_m2 m3s2r1_base(pre-U frozen) vs Master2" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/opp_master2r.py 996000 200 it95base_m2 2>&1 | tee -a $LOG
echo "FIELD base_vs_m2 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
