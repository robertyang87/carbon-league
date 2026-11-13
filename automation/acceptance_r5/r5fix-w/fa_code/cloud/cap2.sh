#!/bin/bash
# platform-equivalent caps on the cloud core (cloud ~0.55x slower than platform budget scale): 0.5/0.75/0.9 s -> 0.28/0.41/0.5
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60 SB=0.28 SH=0.41 SA=0.5
B=../.venv/bin/python
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000 m2ss:cnet2_m2sstand:996000; do IFS=: read n f s <<< "$o"
  rm -f ../data/cap2_stats_$n.jsonl
  CS_STATS=$(pwd)/../data/cap2_stats_$n.jsonl $B bigeval.py exp/cnet2_v7s_mfxc.py exp/$f.py $s 200 c_v7s_mfxc2_$n > ../data/cap2_$n.out 2>&1
done
echo DONE > ../data/cap2.done
