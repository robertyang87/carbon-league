#!/bin/bash
# J1 exam: style rows 2 (Master3s), 3 (M6), 4 (K4TacSearch); pure net, same row all game; vs Master2 real / opp_m3echo / M-style, 200 each
cd /root/carbon/code
until [ -f ../data/weights/cnet2_top_j1L.npz.ok ]; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_top_j1L.npz
for R in 2 3 4; do for O in opp_master2r opp_m3echo cnet2_m2so; do
  AWR_US_CONF="{\"TEACHER\": $R, \"TEACHER_LATE\": $R, \"LATE_STEP\": 240}" EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 j1r${R}_$O > ../data/j1r${R}_$O.out 2>&1 < /dev/null &
done; done
wait; echo DONE > ../data/j1.done
