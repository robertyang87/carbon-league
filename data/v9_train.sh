cd /Users/baggioustb/dev/carbon/carbon-league/code
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128v8.npz BC_LR=1.5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v9 ../data/weights/cnet2_b128v9.npz 3 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128v9.log 2>&1
echo TRAIN_DONE
