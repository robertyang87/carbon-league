cd /Users/baggioustb/dev/carbon/carbon-league/code
rm -rf ../data/cnet2_ds_v13dg3
../.venv/bin/python merge_dagger.py ../data/cnet2_ds_v10 ../data/dagger12,../data/dagger3c ../data/cnet2_ds_v13dg3 3.0 0.2 > ../data/merge_v13dg3.log 2>&1
tail -1 ../data/merge_v13dg3.log
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128dg2.npz BC_LR=1.5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v13dg3 ../data/weights/cnet2_b128dg3.npz 3 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128dg3.log 2>&1
echo TRAIN_DONE
