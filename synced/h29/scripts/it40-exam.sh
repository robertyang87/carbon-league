#!/bin/bash
# ppou2 it=40 checkpoint exam on spot-34 (C1, 2026-11-16) — L20 dispatch 06:30;
# Amendment 3, Echo4 single examiner; pinned set:
# bigeval 354bcd1b / eval_v53 45792dfb / cnet2_search 94f7fb64 / certify_pack fc33be6c.
#  F1 m40_e4:      exp/ppou2_m_it40.py (5460a318) vs packs/echo4_int8.py (819250bf), seed 20298101, 2000
#  F2 m40base_e4:  exp/m3s2r1_base.py (11d80059) vs packs/echo4_int8.py, seed 20298101, 2000
#  F3 m40_e4diag:  exp/ppou2_m_it40.py vs packs/echo4_int8.py, seed 20291101, 1000
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it40_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it40.py packs/echo4_int8.py 20298101 2000 m40_e4
echo "RC_m40_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py packs/echo4_int8.py 20298101 2000 m40base_e4
echo "RC_m40base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it40.py packs/echo4_int8.py 20291101 1000 m40_e4diag
echo "RC_m40_e4diag=$?"
date +%s > /root/carbon/it40_end.txt
echo IT40_ALL_DONE
