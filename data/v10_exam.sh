cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
for net in v10 v9; do for o in m2s2:cnet2_m2s2stand:996000 v13:submission_cnet2_v13:992000 ws:cnet2_wsstand:999500 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  ../.venv/bin/python bigeval.py exp/cnet2_${net}s57_240.py exp/$f.py $s 200 m2_${net}n_$n >> ../data/v10_exam.log 2>&1
done; done
echo V10_EXAM_DONE
