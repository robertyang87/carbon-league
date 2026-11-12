#!/bin/bash
# host 15 (parallel): car>2nd forks | 1-car 50-sim control | camper-bust exam vs 1C-v9 (stalled + normal seeds)
cd /root/carbon/code
for f in fx_m2t3S20n fx_m2t3S50n fx_m2t3S20cb150 fx_m2t3S20cb100; do until [ -f exp/$f.py ]; do sleep 10; done; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
( export AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_b128m2t3.npz AWR_TAU=0
  export AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}' FORK_P=0.3 FORK_H=10,25,50 FORK_END=0.1 FORK_KINDS="car>2nd"
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    FORK_TAG=car_$O ../.venv/bin/python fork_gen.py ../data/fkcar awr_us_env.py exp/$O.py 9200 100 20 > ../data/fkcar_$O.out 2>&1
  done; echo ALL_DONE > ../data/fkcar.done ) &
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S50n.py exp/$O.py 994000 50 sx_S50_$O > ../data/sx_S50_$O.out 2>&1 < /dev/null &
done
for A in S20n S20cb150 S20cb100; do
  EVO_SEEDS=981000,981006,981009,981010,981011,981014,981023,981034,981039,981045,981051,981054,981055,981057,986002,986005,986007,987009,987016,987021,987037,987039,987040,987044,987051,987054,987055,987056,987057,987059,987065,987070,987089,994409,994410,994411,994412,994415,994420,994422,998410,998701,998713,998722,998727,998728,998734,998738,998741,998742,998755,998759,998792,998820,998829,998877,998899 EVO_WORKERS=5 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3$A.py exp/opp_1cz_v9.py 0 0 cbst_$A > ../data/cbst_$A.out 2>&1 < /dev/null &
  EVO_SEEDS=987000,987001,987002,987003,987004,987005,987006,987007,987008,987010,987011,987012,987013,987014,987015,987017,987018,987019,987020,987022,987023,987024,987025,987026,987027,987028,987029,987030,987031,987032 EVO_WORKERS=3 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3$A.py exp/opp_1cz_v9.py 0 0 cbno_$A > ../data/cbno_$A.out 2>&1 < /dev/null &
done
wait
while ps -eo args | grep -q "[b]igeval.py exp/fx_m2t3S50n"; do sleep 30; done; echo S50_DONE > ../data/s50.done
while ps -eo args | grep -q "[o]pp_1cz_v9"; do sleep 30; done; echo CB_DONE > ../data/cb.done
