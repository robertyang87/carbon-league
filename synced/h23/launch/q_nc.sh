#!/bin/bash
# after W5x/W6: deploy 613ed48+ model code, then N-C exam (Echo1+attention, ep9) and E2-Q (multi-car search) redo; T2/T4 later
cd /root/carbon/code
until [ -f ../data/w5x.done ] && [ -f ../data/w6.done ]; do sleep 30; done
cp /root/stage23/cnet2_search.py /root/stage23/cnet2_np.py /root/stage23/cnet2_feat.py ./; md5sum cnet2_search.py cnet2_np.py cnet2_feat.py > ../data/nc.md5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 2, "LATE_STEP": 240}' AWR_US_WEIGHTS=/root/carbon/data/weights/cnet2_m3taL.npz
for O in opp_master2r opp_m3echo cnet2_m2so; do EVO_WORKERS=12 setsid nohup ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 200 nc_$O > ../data/nc_$O.out 2>&1 < /dev/null & done
wait; echo DONE > ../data/nc.done
