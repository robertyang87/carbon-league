#!/bin/bash
# cloud keeps only the 10-sim follow-ups; the 6-sim v13 exam moves to the Mac
pkill -f "cloud/rsq.sh"
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while pgrep -f "bigeval.py exp/cnet2_v7s8" >/dev/null; do sleep 10; done
B=../.venv/bin/python
for t in r10:wb532:opp_wb532:995000 r10:m2ss:cnet2_m2sstand:996000; do IFS=: read r n f s <<< "$t"
  $B bigeval.py exp/cnet2_v7s8_$r.py exp/$f.py $s 200 c_v7s8_${r}_$n >> ../data/rsq.log 2>&1
  echo "$(date +%T) done $r $n" >> ../data/rsq.progress
done
echo "$(date +%T) done r6 m2ss (skipped on cloud)" >> ../data/rsq.progress
