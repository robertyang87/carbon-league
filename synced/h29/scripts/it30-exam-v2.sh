#!/bin/bash
# ppou2 it=30 exam on spot-34 (C1, 2026-11-15) — RESTART under pinned set per strategy
# ruling PPOU2_IT30_HARNESS_RULING.md (2026-11-15 23:11): bigeval 354bcd1b /
# eval_v53 45792dfb / cnet2_search 94f7fb64 / certify_pack fc33be6c.
# Anchor run (bigeval 327ff807) stopped 23:14; its 1453 lines void-preserved on machine
# as data/bigeval_live/void_anchor_m30_e4.jsonl, never delivered, seeds not consumed.
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
