cd /Users/baggioustb/dev/carbon/carbon-league/code
while ! grep -q TRAIN_DONE ../data/v5_train.log; do sleep 30; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=2
../.venv/bin/python bigeval.py exp/cnet2_v5t7g.py exp/submission_cnet2_v13.py 992000 30 v5t7g_v13 &
../.venv/bin/python bigeval.py exp/cnet2_v5t5g.py exp/submission_cnet2_v13.py 992000 30 v5t5g_v13 &
wait
EVO_WORKERS=4 ../.venv/bin/python bigeval.py exp/cnet2_v5t6g.py exp/submission_cnet2_v13.py 992000 30 v5t6g_v13
echo V5_SCREEN_DONE
