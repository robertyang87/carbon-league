#!/bin/bash
cd /root/carbon/code; mv exp/pack_timing.py cloud/ 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
EVO_WORKERS=30 ../.venv/bin/python bigeval.py exp/submission_echo2_w5q8.py exp/opp_master2r.py 994000 200 w5pk_m2r > ../data/w5pk.out 2>&1 &
wait
while [ $(ps -eo args | grep -c "[b]igeval.py") -gt 0 ]; do sleep 20; done
../.venv/bin/python cloud/pack_timing.py exp/submission_echo2_w5q8.py 20 > ../data/w5pk_timing.out 2>&1
echo DONE >> ../data/w5pk_timing.out
