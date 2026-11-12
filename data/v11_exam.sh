cd /Users/baggioustb/dev/carbon/carbon-league/code
while ! grep -q TRAIN_DONE ../data/v11_train.log; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4
for o in m2s2:cnet2_m2s2stand:996000 ws:cnet2_wsstand:999500 v13:submission_cnet2_v13:992000 wb532:opp_wb532:995000; do IFS=: read n f s <<< "$o"
  ../.venv/bin/python bigeval.py exp/cnet2_v11s57_240.py exp/$f.py $s 200 m2_v11n_$n >> ../data/v11_exam.log 2>&1
done
echo V11_EXAM_DONE
