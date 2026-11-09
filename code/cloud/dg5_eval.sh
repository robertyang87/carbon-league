#!/bin/bash
# DGX, after dg5 training: one row per epoch checkpoint so the 6 epochs can be read for monotonicity --
# worst per-slot drop vs dg2 (teacher val, all units), planter heads >1 pt down, held-out agreement on relabelled
# planter decisions (dagger4c val), online probe extra stops (40 games vs the Master2s stand-in; dg2 had 468).
cd ~/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 BC_DEVICE=cuda
until grep -q TRAIN_DONE ../data/cnet2_train_b128dg5.log; do sleep 60; done
W=../data/weights
for e in 0 1 2 3 4 5; do sed "s/dg2/dg5_ep$e/g" exp/cnet2_dg2s57_240.py > exp/cnet2_dg5e${e}s57_240.py; done
for e in 0 1 2 3 4 5; do   # CPU probes, 3 checkpoints at a time x 6 workers
  ../.venv/bin/python dagger_probe.py exp/cnet2_dg5e${e}s57_240.py $W/cnet2_m2v2stand.npz 5 exp/cnet2_m2s2stand.py 996000 40 6 \
    > ../data/dagger_probe_dg5_ep$e.out 2>&1 &
  [ $((e % 3)) -eq 2 ] && wait
done
wait
for e in 0 1 2 3 4 5; do
  ../.venv/bin/python ../data/slot_val.py ../data/cnet2_ds_v10 $W/cnet2_b128dg2.npz $W/cnet2_b128dg5_ep$e.npz > ../data/slot_val_dg5_ep$e.out 2>&1
  ../.venv/bin/python ../data/dg_offline.py ../data/dagger4c $W/cnet2_b128dg5_ep$e.npz > ../data/dg_offline_dg5_ep$e.out 2>&1
done
../.venv/bin/python ../data/dg5_table.py > ../data/dg5_epochs.table 2>&1
echo EVAL_DONE >> ../data/dg5_epochs.table
