#!/bin/bash
# B-class fix test (paired, same maps): v17 vs v17t (threat-first key unit) vs v17te (+ endgame go-home rule).
# Judge on margin and NET cargo (we seize - we lose), not on losses alone; Master2s stand-in only to confirm no regression.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
kill -9 $(pgrep -f "bash /root/pt.sh") 2>/dev/null
while [ ! -f ../data/night.done ]; do sleep 30; done
B=../.venv/bin/python
for v in v17c v17t v17te; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/opp_1cz_v9.py 997000 200 c_${v}_v9 >> ../data/te.log 2>&1; echo "$(date +%T) $v v9" >> ../data/te.progress; done
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  for v in v17t v17te; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/$f.py $s 100 c_${v}_$n >> ../data/te.log 2>&1; echo "$(date +%T) $v $n" >> ../data/te.progress; done
done
$B bigeval.py exp/submission_cnet2_v17te_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17te_m2s2 >> ../data/te.log 2>&1; echo "$(date +%T) v17te m2s2" >> ../data/te.progress
echo DONE > ../data/te.done
