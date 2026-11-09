cd /Users/baggioustb/dev/carbon/carbon-league/code
until grep -q TRAIN_DONE /private/tmp/claude-501/-Users-baggioustb-dev-carbon/7b471ad7-6df8-4bd4-8823-f8d4ab083fc6/tasks/btpj6teqb.output 2>/dev/null; do sleep 30; done
grep '"val"' ../data/cnet2_train_b128dg3.log | cut -c1-200
for d in ../data/dagger12 ../data/dagger3c; do ../.venv/bin/python ../data/dg_offline.py $d ../data/weights/cnet2_b128dg2.npz ../data/weights/cnet2_b128dg3.npz; done
../.venv/bin/python dagger_probe.py exp/cnet2_dg3s57_240.py ../data/weights/cnet2_m2v2stand.npz 5 exp/cnet2_m2s2stand.py 996000 40 4 > ../data/dagger_probe_dg3.out 2>&1
cat ../data/dagger_probe_dg3.out
