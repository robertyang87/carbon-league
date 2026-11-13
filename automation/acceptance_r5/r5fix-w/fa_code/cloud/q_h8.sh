#!/bin/bash
# host 8 (2026-11-10 night): SHARE_OPP level 2 vs off, both cap 6 / uniform / full 30 sims, head-to-head
cd /root/carbon/code; export OMP_NUM_THREADS=1
EVO_WORKERS=62 setsid nohup ../.venv/bin/python bigeval.py exp/fx_U6S2.py exp/fx_U6S0.py 983000 400 sh_S2_vs_S0_h8 > ../data/sh_h8.out 2>&1 < /dev/null &
wait
