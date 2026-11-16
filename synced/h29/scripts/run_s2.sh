#!/bin/bash
# S2 (RULING_R2): deployment-timing evidence for the locked search package
# echo4_search_int8.py (be96d457) vs Master2 (opp_master2r.py 2aebfbc1).
# Diagnostic-grade timing evidence only; NOT a production timing certification.
# certify_pack fc33be6c, systemd scope CPUQuota=150% + MemoryMax=512M,
# --require-quota + --mem-limit-mib 512, BLAS pinned to 1. Seeds 986500-986519,
# 20 games, both seats (side = seed % 2 inside certify).
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
systemd-run --scope -p CPUQuota=150% -p MemoryMax=512M --unit=s2-echo4search-cost /root/carbon/.venv/bin/python certify_pack.py packs/echo4_search_int8.py --opp exp/opp_master2r.py --seed0 986500 --games 20 --require-quota --mem-limit-mib 512 --out /root/carbon/s2_echo4search_cost.json
echo "CERTIFY_RC=$?" >> /root/carbon/s2_run.log
