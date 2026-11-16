#!/bin/bash
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
systemd-run --scope -p CPUQuota=150% -p MemoryMax=512M --unit=s0a-echo4-cost /root/carbon/.venv/bin/python certify_pack.py packs/echo4_int8.py --opp exp/opp_master2r.py --seed0 993000 --games 20 --require-quota --mem-limit-mib 512 --out /root/carbon/s0a_echo4_cost.json
echo "CERTIFY_RC=$?" >> /root/carbon/s0a_run.log
