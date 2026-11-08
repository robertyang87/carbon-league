#!/bin/bash
# v17 exam baselines = the online v16b. On this oversubscribed box the clock caps would truncate the search, so the
# baseline uses the identical build without caps (submission_cnet2_v16s_mixnc.py: same nets, same quantization).
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/foreign.done ]; do sleep 20; done
for o in m2s2:cnet2_m2s2stand:996000 ws:cnet2_wsstand:999500; do IFS=: read n f s <<< "$o"
  ../.venv/bin/python bigeval.py exp/submission_cnet2_v16s_mixnc.py exp/$f.py $s 200 c_v16b_$n >> ../data/v16bbase.log 2>&1
done
echo DONE > ../data/v16bbase.done
