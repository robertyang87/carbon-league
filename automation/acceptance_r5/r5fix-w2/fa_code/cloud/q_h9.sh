#!/bin/bash
# host 9 (2026-11-10 night): step-3 temperature samples vs the Grove1 net, pick tau by the runbook rule, then sp1_grove1 (2500 whole games)
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_STEPS=300 AWR_OPEN_STEPS=300 AWR_KINDS=planter,base
export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz
export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
for T in 0.3 0.5 0.8; do ../.venv/bin/python awr_gen.py ../data/sp_tau$T awr_us_env.py exp/g1_net.py 3100000 40 4 $T 62 > ../data/sp_tau$T.out 2>&1; done
TAU=$(python3 - <<'PY'
import json
best = None
for T in ("0.3", "0.5", "0.8"):
    try: r = json.loads(open(f"/root/carbon/data/sp_tau{T}.out").read().strip().splitlines()[-1])
    except Exception: continue
    if r["adv_mean"] >= -150 and (best is None or r["adv_sd"] > best[1]): best = (T, r["adv_sd"])
print(best[0] if best else "0.3")
PY
)
echo "chosen tau $TAU" > ../data/sp_tau_chosen.txt
../.venv/bin/python awr_gen.py ../data/sp1_grove1 awr_us_env.py exp/g1_net.py 3000000 500 4 $TAU 62 > ../data/sp1_grove1.out 2>&1
