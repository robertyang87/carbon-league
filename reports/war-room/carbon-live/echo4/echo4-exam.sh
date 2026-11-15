#!/bin/bash
# Echo4 launch exam on spot-34 (C1, 2026-11-15)
#  F1 e4pack_m2: packs/echo4_int8.py (819250bf) vs exp/opp_master2r.py (2aebfbc1), seed 996000, 200
#  F2 e4bare_m2: exp/ppou2_m_it20.py (f4fff163) vs exp/opp_master2r.py, seed 996000, 200
#  F3 e4pack_e2: packs/echo4_int8.py vs exp/echo2.py (15b6d754), seed 999000, 200
cd /root/carbon || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
date +%s > /root/carbon/echo4_start.txt
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_int8.py exp/opp_master2r.py 996000 200 e4pack_m2
echo "RC_pack_m2=$?"
/root/carbon/.venv/bin/python code/bigeval.py exp/ppou2_m_it20.py exp/opp_master2r.py 996000 200 e4bare_m2
echo "RC_bare_m2=$?"
/root/carbon/.venv/bin/python code/bigeval.py packs/echo4_int8.py exp/echo2.py 999000 200 e4pack_e2
echo "RC_pack_e2=$?"
date +%s > /root/carbon/echo4_end.txt
echo ECHO4_ALL_DONE
