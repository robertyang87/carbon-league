#!/bin/bash
# Win-rate screen (REGISTRATION_WINRATE_SCREEN_20261116, strategy group):
# search pack vs no-search baseline, direct paired, 600 seeds x 2 seats.
# Runs ON spot-34, cwd=/root/carbon. Pinned harness set only. CONF verbatim.
#  F_A screen_search_a: packs/echo4_search_int8.py (A) vs packs/echo4_int8.py, seed 989200, 600
#  F_B screen_search_b: packs/echo4_int8.py (A) vs packs/echo4_search_int8.py, seed 989200, 600
# Per seed, the search pack sits seat (seed%2) in F_A and seat (1-seed%2) in F_B:
# one seat-0 game and one seat-1 game per seed, 1,200 games total.
cd /root/carbon || exit 1
export EVO_WORKERS=62
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/screen_start.txt
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_search_int8.py packs/echo4_int8.py 989200 600 screen_search_a
echo "RC_screen_search_a=$?"
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_int8.py packs/echo4_search_int8.py 989200 600 screen_search_b
echo "RC_screen_search_b=$?"
date +%s > /root/carbon/screen_end.txt
echo SCREEN_ALL_DONE
