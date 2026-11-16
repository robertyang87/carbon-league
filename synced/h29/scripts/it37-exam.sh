#!/bin/bash
# ppou2 it=37 cap exam on spot-34 (C1, 2026-11-16) — L20 dispatch 02:44; pinned set:
# bigeval 354bcd1b / eval_v53 45792dfb / cnet2_search 94f7fb64 / certify_pack fc33be6c.
# Anchor a7ed8f95 never enters the exam field.
#  F1 m37_e2:      exp/ppou2_m_it37.py (10a6b2de) vs exp/echo2.py (15b6d754), seed 982000, 2000
#  F2 m37base_e2:  exp/m3s2r1_base.py (11d80059) vs exp/echo2.py, seed 982000, 2000
#  F3 m37_e4:      exp/ppou2_m_it37.py vs packs/echo4_int8.py (819250bf), seed 20296101, 2000
#  F4 m37base_e4:  exp/m3s2r1_base.py vs packs/echo4_int8.py, seed 20296101, 2000
#  F5 m37_e4diag:  exp/ppou2_m_it37.py vs packs/echo4_int8.py, seed 20291101, 1000
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it37_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it37.py exp/echo2.py 982000 2000 m37_e2
echo "RC_m37_e2=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py exp/echo2.py 982000 2000 m37base_e2
echo "RC_m37base_e2=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it37.py packs/echo4_int8.py 20296101 2000 m37_e4
echo "RC_m37_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py packs/echo4_int8.py 20296101 2000 m37base_e4
echo "RC_m37base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it37.py packs/echo4_int8.py 20291101 1000 m37_e4diag
echo "RC_m37_e4diag=$?"
date +%s > /root/carbon/it37_end.txt
echo IT37_ALL_DONE
