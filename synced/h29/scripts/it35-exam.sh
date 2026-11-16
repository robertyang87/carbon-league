#!/bin/bash
# ppou2 it=35 exam on spot-34 (C1, 2026-11-16) — pinned set per PPOU2_IT35_TRIGGERS.md:
# bigeval 354bcd1b / eval_v53 45792dfb / cnet2_search 94f7fb64 / certify_pack fc33be6c.
# Anchor a7ed8f95 never enters the exam field.
#  F1 m35_e4:      exp/ppou2_m_it35.py (88c43c47) vs packs/echo4_int8.py (819250bf), seed 20294101, 2000
#  F2 m35base_e4:  exp/m3s2r1_base.py (11d80059) vs packs/echo4_int8.py, seed 20294101, 2000
#  F3 m35_e4diag:  exp/ppou2_m_it35.py vs packs/echo4_int8.py, seed 20291101, 1000
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it35_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it35.py packs/echo4_int8.py 20294101 2000 m35_e4
echo "RC_m35_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py packs/echo4_int8.py 20294101 2000 m35base_e4
echo "RC_m35base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it35.py packs/echo4_int8.py 20291101 1000 m35_e4diag
echo "RC_m35_e4diag=$?"
date +%s > /root/carbon/it35_end.txt
echo IT35_ALL_DONE
