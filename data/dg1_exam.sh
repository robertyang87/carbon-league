cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64
# 1) the two causal metrics first: stop ratio on our own states (probe), then planter ledger / tree life
../.venv/bin/python dagger_probe.py exp/cnet2_dg1s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 exp/cnet2_m2s2stand.py 996000 40 4 > ../data/dagger_probe_dg1.out 2>&1
echo PROBE_DONE
export EVO_WORKERS=4
for o in m2s2:cnet2_m2s2stand:996000 ws:cnet2_wsstand:999500 v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  ../.venv/bin/python bigeval.py exp/cnet2_dg1s57_240.py exp/$f.py $s 200 m2_dg1n_$n >> ../data/dg1_exam.log 2>&1
done
echo DG1_EXAM_DONE
