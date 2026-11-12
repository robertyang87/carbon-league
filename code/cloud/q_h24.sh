#!/bin/bash
# host 24: N-C exam, N0-replica exam (when shipped), camp-bust step 1 (E0 vs 1C-v9). All pure net, row 2 all game, seeds 994000.
cd /root/carbon/code
until [ -f exp/master2_src.py ]; do sleep 20; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}'
ex() { W=$1; T=$2; shift 2; for O in "$@"; do AWR_US_WEIGHTS=/root/carbon/data/weights/$W.npz EVO_WORKERS=10 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 ${T}_$O > ../data/${T}_$O.out 2>&1 < /dev/null & done; }
ex cnet2_m3w3L cb1c_E0 opp_1cz_v9
( until [ -f ../data/weights/cnet2_m3n0L.npz.ok ]; do sleep 30; done; ex cnet2_m3n0L n0r opp_master2r opp_m3echo cnet2_m2so; wait; echo DONE > ../data/n0r.done ) &
wait; echo DONE > ../data/h24a.done
