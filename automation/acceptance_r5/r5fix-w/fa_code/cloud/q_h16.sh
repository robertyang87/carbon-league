#!/bin/bash
# host 16: second half of the 1-car 50-sim control (seeds 994050-994099), so it matches the 100/opponent multi-car exam
cd /root/carbon/code
until [ -f exp/fx_m2t3S50n.py ]; do sleep 10; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for O in cnet2_m2s2stand cnet2_m2so s1_net; do
  EVO_WORKERS=21 setsid nohup ../.venv/bin/python bigeval.py exp/fx_m2t3S50n.py exp/$O.py 994050 50 sx_S50b_$O > ../data/sx_S50b_$O.out 2>&1 < /dev/null &
done
wait; echo S50B_DONE > ../data/s50b.done
