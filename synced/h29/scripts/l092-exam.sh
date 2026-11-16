#!/bin/bash
# ppou2 l092 it=20 exam on spot-34 (C1, 2026-11-16) — L20 dispatch 07:34;
# Amendment 3; pinned set:
# bigeval 354bcd1b / eval_v53 45792dfb / cnet2_search 94f7fb64 / certify_pack fc33be6c.
#  F1 l092_it20_m2:     exp/ppou2_l092_it20.py (013b2b68) vs exp/opp_master2r.py (2aebfbc1), seed 980000, 2000
#  F2 l092_it20_e4:     exp/ppou2_l092_it20.py vs packs/echo4_int8.py (819250bf), seed 20300101, 2000
#  F3 l092base_e4:      exp/m3s2r1_base.py (11d80059) vs packs/echo4_int8.py, seed 20300101, 2000
#  F4 l092_it20_e4diag: exp/ppou2_l092_it20.py vs packs/echo4_int8.py, seed 20291101, 1000
#  (No F5: Master2 leg pairs against frozen baseline ppou2_base_m2_a.jsonl
#   md5 777fc1ac, seeds 980000-981999, mean U 0.8242 — L20 07:39, do not re-run.)
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/l092_start.txt
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_l092_it20.py exp/opp_master2r.py 980000 2000 l092_it20_m2
echo "RC_l092_it20_m2=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_l092_it20.py packs/echo4_int8.py 20300101 2000 l092_it20_e4
echo "RC_l092_it20_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1_base.py packs/echo4_int8.py 20300101 2000 l092base_e4
echo "RC_l092base_e4=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_l092_it20.py packs/echo4_int8.py 20291101 1000 l092_it20_e4diag
echo "RC_l092_it20_e4diag=$?"
date +%s > /root/carbon/l092_end.txt
echo L092_ALL_DONE
