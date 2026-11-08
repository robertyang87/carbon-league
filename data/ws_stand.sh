cd /Users/baggioustb/dev/carbon/carbon-league/code
while pgrep -f "ladder_corpus.py ../data/wushi_corpus" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 BUILD_WORKERS=4 ../.venv/bin/python build_ws.py ../data/cnet2_ds_ws ../data/wushi_corpus > ../data/cnet2_build_ws.log 2>&1
tail -1 ../data/cnet2_build_ws.log
BC_INIT=../data/weights/cnet2_b128.npz BC_LR=2e-4 BC_BS=256 BC_AMP=fp32 BC_DEVICE=mps ../.venv/bin/python cnet2_train.py ../data/cnet2_ds_ws ../data/weights/cnet2_wsstand.npz 4 '{"C":128,"NB":8,"D":128}' > ../data/cnet2_train_wsstand.log 2>&1
echo WS_DONE
