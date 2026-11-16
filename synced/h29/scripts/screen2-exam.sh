#!/bin/bash
# Winrate screen ROUND 2 (successor kernel): resync search pack vs no-search baseline.
# Registration: reports/search_dossier/REGISTRATION_WINRATE_SCREEN_R2_RESYNC_20261116.md @bda12123
# Runs ON spot-34, cwd=/root/carbon. Pinned harness set only.
# Serial:
#  A screen_r2_a: packs/echo4_search_int8_resync.py (cba5923c) vs packs/echo4_int8.py (819250bf), seed 998400, 600
#  B screen_r2_b: packs/echo4_int8.py vs packs/echo4_search_int8_resync.py, seed 998400, 600
cd /root/carbon || exit 1
export EVO_WORKERS=62
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/screen2_start.txt
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_search_int8_resync.py packs/echo4_int8.py 998400 600 screen_r2_a
echo "RC_screen_r2_a=$?"
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_int8.py packs/echo4_search_int8_resync.py 998400 600 screen_r2_b
echo "RC_screen_r2_b=$?"
date +%s > /root/carbon/screen2_end.txt
echo SCREEN2_ALL_DONE
