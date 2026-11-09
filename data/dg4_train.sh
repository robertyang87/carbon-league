cd /Users/baggioustb/dev/carbon/carbon-league/code
# wait for both fixed-prefix DAgger sets (1000 games each) to be backed up from the cloud
until [ $(ls ../data/dagger4a/shards 2>/dev/null | wc -l) -ge 1000 ] && [ $(ls ../data/dagger4c/shards 2>/dev/null | wc -l) -ge 1000 ] \
      && [ $(wc -l < ../data/dagger4c/games.jsonl 2>/dev/null || echo 0) -ge 1000 ]; do sleep 60; done

rm -rf ../data/cnet2_ds_v14dg4
../.venv/bin/python merge_dagger.py ../data/cnet2_ds_v10 ../data/dagger4a,../data/dagger4c ../data/cnet2_ds_v14dg4 3.0 1.0   # dg2 mix: all teacher data, no slot weights > ../data/merge_v14dg4.log 2>&1
tail -1 ../data/merge_v14dg4.log
BC_DEVICE=mps BC_INIT=../data/weights/cnet2_b128dg2.npz BC_LR=1.5e-4 ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_v14dg4 ../data/weights/cnet2_b128dg4.npz 2 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_b128dg4.log 2>&1
echo TRAIN_DONE
sed 's/dg2/dg4/g' exp/cnet2_dg2s57_240.py > exp/cnet2_dg4s57_240.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64
../.venv/bin/python ../data/slot_val.py ../data/cnet2_ds_v10 ../data/weights/cnet2_b128dg2.npz ../data/weights/cnet2_b128dg4.npz > ../data/slot_val_dg4.out 2>&1; tail -1 ../data/slot_val_dg4.out
for d in ../data/dagger4a ../data/dagger4c; do ../.venv/bin/python ../data/dg_offline.py $d ../data/weights/cnet2_b128dg2.npz ../data/weights/cnet2_b128dg4.npz; done
../.venv/bin/python dagger_probe.py exp/cnet2_dg4s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 exp/cnet2_m2s2stand.py 996000 40 4 > ../data/dagger_probe_dg4.out 2>&1; cat ../data/dagger_probe_dg4.out
EVO_WORKERS=4 ../.venv/bin/python bigeval.py exp/cnet2_dg4s57_240.py exp/cnet2_m2s2stand.py 996000 200 m2_dg4n_m2s2 >> ../data/dg4_tree.log 2>&1
python3 ../data/planter_check.py m2_v9n_m2s2 m2_dg2n_m2s2 m2_dg3n_m2s2 m2_dg4n_m2s2
python3 ../data/paired.py m2_dg4n_m2s2 m2_dg2n_m2s2 | tail -1; python3 ../data/paired.py m2_dg4n_m2s2 m2_v9n_m2s2 | tail -1
