#!/bin/bash
cd /root/carbon
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG=/root/carbon/data/probe-exam.log
echo "START $(date '+%F %T %Z') field it95_vs_g1 it95(snipe_n1 it95 frozen) vs m3s2g1" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/it95.py exp/m3s2g1_base.py 990000 200 probe_it95_g1 2>&1 | tee -a $LOG
echo "FIELD it95_vs_g1 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "START $(date '+%F %T %Z') field base_vs_g1 m3s2r1_base(pre-U frozen) vs m3s2g1" | tee -a $LOG
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/m3s2g1_base.py 990000 200 probe_base_g1 2>&1 | tee -a $LOG
echo "FIELD base_vs_g1 DONE $(date '+%F %T %Z')" | tee -a $LOG
echo "ALL DONE $(date '+%F %T %Z')" | tee -a $LOG
