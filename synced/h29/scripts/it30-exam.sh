#!/bin/bash
# ppou2 it=30 exam on spot-34 (C1, 2026-11-15) — anchor a7ed8f95 harness bigeval 327ff807
#  F1 m30_e4:     exp/ppou2_m_it30.py (79331bfe) vs packs/echo4_int8.py (819250bf), seed 20292101, 2000
#  F2 base_e4:    exp/m3s2r1_base.py (11d80059) vs packs/echo4_int8.py, seed 20292101, 2000
#  F3 m30_e4diag: exp/ppou2_m_it30.py vs packs/echo4_int8.py, seed 20291101, 1000
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it30_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it30.py packs/echo4_int8.py 20292101 2000 m30_e4
echo "RC_m30_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py packs/echo4_int8.py 20292101 2000 base_e4
echo "RC_base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it30.py packs/echo4_int8.py 20291101 1000 m30_e4diag
echo "RC_m30_e4diag=$?"
date +%s > /root/carbon/it30_end.txt
echo IT30_ALL_DONE
