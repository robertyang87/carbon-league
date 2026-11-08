#!/bin/bash
# priority queue: the decision exam (v13) for 10 and 6 sims first, then WB-532 / Master2s stand-in
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
B=../.venv/bin/python
for t in r10:v13:submission_cnet2_v13:992000 r6:v13:submission_cnet2_v13:992000 r10:wb532:opp_wb532:995000 r10:m2ss:cnet2_m2sstand:996000 r6:wb532:opp_wb532:995000 r6:m2ss:cnet2_m2sstand:996000; do
  IFS=: read r n f s <<< "$t"
  $B bigeval.py exp/cnet2_v7s_mfxn8_$r.py exp/$f.py $s 200 c_v7s8_${r}_$n >> ../data/rsq.log 2>&1
  echo "$(date +%T) done $r $n" >> ../data/rsq.progress
done
echo DONE > ../data/rsq.done
