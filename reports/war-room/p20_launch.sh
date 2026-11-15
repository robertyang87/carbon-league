#!/bin/bash
# C2 diagnostic field: ppou2 it20 pack vs Echo3 deployment pack, spot-34.
# 800 seeds (985200-985999) x double seat = 1600 games. Sequential A then B.
set -u
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
date +%s > /root/carbon/data/p20_start.txt
$PY bigeval.py exp/ppou2_m_it20.py exp/echo3_int8.py 985200 800 p20_a > /root/carbon/data/p20_a.launcher.log 2> /root/carbon/data/p20_a.err.log
echo "RC_a=$?" >> /root/carbon/data/p20_end.txt
$PY bigeval.py exp/echo3_int8.py exp/ppou2_m_it20.py 985200 800 p20_b > /root/carbon/data/p20_b.launcher.log 2> /root/carbon/data/p20_b.err.log
echo "RC_b=$?" >> /root/carbon/data/p20_end.txt
echo "P20_ALL_DONE" >> /root/carbon/data/p20_end.txt
date +%s >> /root/carbon/data/p20_end.txt
