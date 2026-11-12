#!/bin/bash
# escape priority test: v17 vs v17 + PRIO_THREAT 2 (threatened loaded car outranks hunting targets for the one key slot)
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/night.done ]; do sleep 30; done
B=../.venv/bin/python
for v in v17c v17pt; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/opp_1cz_v9.py 997000 200 c_${v}_v9 >> ../data/pt.log 2>&1; echo "$(date +%T) $v v9" >> ../data/pt.progress; done
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/submission_cnet2_v17pt_nc.py exp/$f.py $s 100 c_v17pt_$n >> ../data/pt.log 2>&1; echo "$(date +%T) v17pt $n" >> ../data/pt.progress; done
echo DONE > ../data/pt.done
