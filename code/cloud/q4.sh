#!/bin/bash
# New host (139.196.142.134) queue, 2026-11-09. All paired on fixed seeds; baselines c_v17c_* (old host, bit-identical).
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=60
B=../.venv/bin/python; P=../data/q4.progress
step() { echo "$(date +%T) $1" >> $P; }
# A) dg4 DAgger data on dg2's own states (fixed prefix), half default pool / half 1C-style own nets
mkdir -p ../data/dagger4a ../data/dagger4c
$B dagger_gen.py ../data/dagger4a exp/cnet2_dg2s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 20000 1000 60 > ../data/dagger4a.out 2>&1; step "dagger4a"
DG_POOL=exp/cnet2_v5t7g.py,exp/cnet2_v6t7g.py,exp/cnet2_v6t6g.py,exp/cnet2_v9s57_240.py \
  $B dagger_gen.py ../data/dagger4c exp/cnet2_dg2s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 30000 1000 60 > ../data/dagger4c.out 2>&1; step "dagger4c"
# B) dg2 policy inside the v17 search (pure-net exam: +205 +- 80 over v9 on the Master2s stand-in)
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17d2_m2s2 >> ../data/q4.log 2>&1; step "v17d2 m2s2"
$B bigeval.py exp/submission_cnet2_v17c_nc.py exp/opp_1cz_v9.py 997000 200 c_v17c_v9 >> ../data/q4.log 2>&1; step "v17c v9"
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/opp_1cz_v9.py 997000 200 c_v17d2_v9 >> ../data/q4.log 2>&1; step "v17d2 v9"
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  $B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/$f.py $s 100 c_v17d2_$n >> ../data/q4.log 2>&1; step "v17d2 $n"; done
# C) endgame rules (te.sh): judge on margin, strand, enemy-seized late30 and late30_eh vs the v17t shadow
for v in v17t v17te v17tb; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/opp_1cz_v9.py 997000 200 c_${v}_v9 >> ../data/q4.log 2>&1; step "$v v9"; done
for o in v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  for v in v17t v17te; do $B bigeval.py exp/submission_cnet2_${v}_nc.py exp/$f.py $s 100 c_${v}_$n >> ../data/q4.log 2>&1; step "$v $n"; done; done
$B bigeval.py exp/submission_cnet2_v17te_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17te_m2s2 >> ../data/q4.log 2>&1; step "v17te m2s2"
echo DONE > ../data/q4.done
