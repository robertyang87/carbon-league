cd /Users/baggioustb/dev/carbon/carbon-league/code
until grep -q TRAIN_DONE /private/tmp/claude-501/-Users-baggioustb-dev-carbon/7b471ad7-6df8-4bd4-8823-f8d4ab083fc6/tasks/btpj6teqb.output 2>/dev/null; do sleep 30; done
../.venv/bin/python ../data/slot_val.py ../data/cnet2_ds_v10 ../data/weights/cnet2_b128dg2.npz ../data/weights/cnet2_b128dg3.npz > ../data/slot_val_dg3.out 2>&1
tail -1 ../data/slot_val_dg3.out
