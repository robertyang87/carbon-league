#!/bin/bash
# ppou2 main arm it=50 checkpoint exam (owner single-field rule 2026-11-16).
# Registration: reports/PPOU2_IT50_EXAM_REGISTRATION_20261116.md @055347c4
# Single field only:
#  m50_e4: exp/ppou2_m_it50.py (c35305af) vs packs/echo4_int8.py (819250bf), seed 20304101, 2000
# Runs ON spot-34, cwd=/root/carbon. Pinned harness set only.
cd /root/carbon || exit 1
export EVO_WORKERS=62
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/it50_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it50.py packs/echo4_int8.py 20304101 2000 m50_e4
echo "RC_m50_e4=$?"
date +%s > /root/carbon/it50_end.txt
echo IT50_ALL_DONE
