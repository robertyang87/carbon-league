#!/bin/bash
# runbook_echo2 3b: ensembles on Echo1 (W3L) vs Master2 real and M-style, 200 each (pairs with plm_E0_*), then 10-game timing each
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3w3L.npz
W=/root/carbon/data/weights
declare -A E=( [EN2]="[\"$W/cnet2_m3w1L.npz\"]" [EN3]="[\"$W/cnet2_m3w1L.npz\", \"$W/cnet2_m3w2L.npz\"]" [EN4]="[\"$W/cnet2_m3w1L.npz\", \"$W/cnet2_m3w3b.npz\"]" )
for v in EN2 EN3 EN4; do for O in opp_master2r cnet2_m2so; do
  AWR_US_CONF="{\"TEACHER\": 2, \"TEACHER_LATE\": 2, \"LATE_STEP\": 240, \"ENS\": ${E[$v]}}" EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ens_${v}_$O > ../data/ens_${v}_$O.out 2>&1 < /dev/null &
done; done
wait
for v in EN2 EN3 EN4; do AWR_US_CONF="{\"TEACHER\": 2, \"TEACHER_LATE\": 2, \"LATE_STEP\": 240, \"ENS\": ${E[$v]}}" ../.venv/bin/python cloud/pack_timing.py awr_us_env.py 10 | sed "s/^/$v /" >> ../data/ens_timing.out 2>&1; done
echo DONE > ../data/ens.done
