cd /Users/baggioustb/dev/carbon/carbon-league/code
rm -rf ../data/cnet2_ds_v12dg
../.venv/bin/python merge_dagger.py ../data/cnet2_ds_v10 ../data/dagger1 ../data/cnet2_ds_v12dg 1.0 > ../data/merge_v12dg.log 2>&1
tail -1 ../data/merge_v12dg.log
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128v10.npz BC_LR=1.5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v12dg ../data/weights/cnet2_b128dg1.npz 2 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128dg1.log 2>&1
echo TRAIN_DONE
