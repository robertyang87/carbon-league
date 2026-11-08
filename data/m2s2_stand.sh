cd /Users/baggioustb/dev/carbon/carbon-league/code
while pgrep -f "ladder_corpus.py ../data/m2s_corpus" >/dev/null; do sleep 15; done
tail -1 ../data/m2s_corpus.log
rm -rf ../data/cnet2_ds_m2s2
OMP_NUM_THREADS=1 BUILD_WORKERS=4 ../.venv/bin/python build_m2s2.py ../data/cnet2_ds_m2s2 ../data/m2s_corpus > ../data/cnet2_build_m2s2.log 2>&1
tail -1 ../data/cnet2_build_m2s2.log
BC_INIT=../data/weights/cnet2_b128v7.npz BC_LR=2e-4 BC_BS=256 BC_AMP=fp32 BC_DEVICE=mps ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_m2s2 ../data/weights/cnet2_m2s2stand.npz 4 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_m2s2stand.log 2>&1
grep '"val"' ../data/cnet2_train_m2s2stand.log | tail -1
echo M2S2_DONE
