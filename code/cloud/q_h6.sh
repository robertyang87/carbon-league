#!/bin/bash
# host 6 (2026-11-10): phase-2 budget comparisons, fixed budget, head-to-head
cd /root/carbon/code; export OMP_NUM_THREADS=1
r() { EVO_WORKERS=$4 setsid nohup ../.venv/bin/python bigeval.py exp/$1.py exp/$2.py 992500 200 $3 > ../data/$3.out 2>&1 < /dev/null & }
r fx_m2t3RS7 fx_m2t3S20 p2_RS7_vs_S20 13
r fx_m2t3RS6 fx_m2t3S20 p2_RS6_vs_S20 12
r fx_m2t3RS8 fx_m2t3S20 p2_RS8_vs_S20 13
r fx_m2t3S20H5 fx_m2t3S40 p2_H5x20_vs_H3x40 13
r fx_m2t3S40BrS fx_m2t3S40 p2_BrS40_vs_G40 13
wait
