#!/bin/bash
# overnight queue (keeps the box busy to ~08:00+): after q3 -> DAgger drift probe, then v19 candidates
# v19a = b128v10 + current search; v19b = b128v10 + v18b endgame config. Paired vs v16b / v17 on the same maps.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
while [ ! -f ../data/q2.done ]; do sleep 20; done
B=../.venv/bin/python; P=../data/night.progress
EVO_WORKERS=1 $B -c "print(1)" >/dev/null
EVO_WORKERS=60 $B bigeval.py exp/submission_cnet2_v18c_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v18c_m2s2 >> ../data/night.log 2>&1; echo "$(date +%T) v18c m2s2" >> $P
EVO_WORKERS=60 $B bigeval.py exp/submission_cnet2_v18c_nc.py exp/cnet2_wsstand.py 999500 100 c_v18c_ws >> ../data/night.log 2>&1; echo "$(date +%T) v18c ws" >> $P
$B dagger_probe.py exp/cnet2_v9s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 exp/submission_cnet2_v15.py 996000 40 40 > ../data/dagger_probe_v15.out 2>&1; echo "$(date +%T) probe" >> $P
export EVO_WORKERS=60
for v in v19a v19b; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/cnet2_m2s2stand.py 996000 200 c_${v}_m2s2 >> ../data/night.log 2>&1; echo "$(date +%T) $v m2s2" >> $P; done
for o in ws:cnet2_wsstand:999500 v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  for v in v19a v19b; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/$f.py $s 100 c_${v}_$n >> ../data/night.log 2>&1; echo "$(date +%T) $v $n" >> $P; done
done
# tail filler (only if still early): more Master2s stand-in games for the two v19 candidates on fresh maps
for v in v19a v19b; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/cnet2_m2s2stand.py 996200 200 c_${v}_m2s2b >> ../data/night.log 2>&1; echo "$(date +%T) $v m2s2b" >> $P; done
echo DONE > ../data/night.done
