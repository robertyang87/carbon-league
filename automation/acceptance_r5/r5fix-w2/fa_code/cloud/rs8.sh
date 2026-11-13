#!/bin/bash
# 8-style opponent model + v7 rollout net, fixed 4/6 sims; waits for rs.sh, then installs the new eval code
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/rs.done ]; do sleep 20; done
cp .cnet2_search_next.py cnet2_search.py; cp .eval_v53_next.py eval_v53.py; cp .bigeval_next.py bigeval.py
B=../.venv/bin/python
for r in 6 4; do for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000 m2ss:cnet2_m2sstand:996000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/cnet2_v7s_mfxn8_r$r.py exp/$f.py $s 200 c_v7s8_r${r}_$n >> ../data/rs8.log 2>&1
done; done
echo DONE > ../data/rs8.done
