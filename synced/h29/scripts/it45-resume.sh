#!/bin/bash
# ppou2 it=45 checkpoint exam RESUME after winrate-screen preemption.
# F1 m45_e4 already complete and kept (2000 rows). This script runs only:
#  F2 m45base_e4:  exp/ppou2_m_it40.py vs packs/echo4_int8.py, seed 20302101, 2000 (full rerun; the 374-row partial is VOID_screen_preempt, never merged)
#  F3 m45_e4diag:  exp/ppou2_m_it45.py vs packs/echo4_int8.py, seed 20291101, 1000
# Runs ON spot-34, cwd=/root/carbon. Pinned harness set only.
cd /root/carbon || exit 1
export EVO_WORKERS=62
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it45_resume_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it40.py packs/echo4_int8.py 20302101 2000 m45base_e4
echo "RC_m45base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it45.py packs/echo4_int8.py 20291101 1000 m45_e4diag
echo "RC_m45_e4diag=$?"
date +%s > /root/carbon/it45_end.txt
echo IT45_ALL_DONE
