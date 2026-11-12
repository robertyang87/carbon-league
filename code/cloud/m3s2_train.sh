#!/bin/zsh
# Rebuild Master3s dataset from the refreshed corpus, train 3 planter/base mixes (decoder-only, KL30, 10 epochs).
cd /Users/baggioustb/dev/carbon/carbon-league/code
D=../data; I=$D/top_corpus/index.json; t0=$(stat -f %m $I)
for i in {1..60}; do [ $(stat -f %m $I) -gt $t0 ] && break; sleep 10; done
../.venv/bin/python - <<'PY'
import json,os
ix=json.load(open('../data/top_corpus/index.json'))
m=[g for g in ix if any(s.get('name')=='claude-AlphaCarbon-Master3s' for s in g['strategies'].values())]
for g in m:
    d='../data/m3s_corpus/'+g['file']
    if not os.path.lexists(d): os.symlink(os.path.abspath('../data/top_corpus/'+g['file']),d)
json.dump(m,open('../data/m3s_corpus/index.json','w')); print('m3s games',len(m))
PY
rm -rf $D/cnet2_ds_m3s2
../.venv/bin/python build_m3s.py $D/cnet2_ds_m3s2 $D/m3s_corpus || exit 1
for v in a:0:1:0 b:0:1:0.3 c:0:1:1; do n=${v%%:*}; r=${v#*:}; X=$D/cnet2_ds_m3s2$n
  rm -rf $X; mkdir $X; for f in $D/cnet2_ds_m3s2/*; do ln -s ../cnet2_ds_m3s2/${f:t} $X/${f:t}; done; rm -f $X/uw.npy $X/w.npy
  ../.venv/bin/python set_uw.py $X ${(s/:/)r}
done
for n in a b c; do
  env BC_DEVICE=cpu BC_RAM=1 BC_THREADS=4 BC_SAVE_EPOCHS=0 BC_INIT=$D/weights/cnet2_b128m2t3.npz BC_TRAIN_PREFIX=teach,din,aemb,gru,whead,bhead,U,V,h0 BC_TEACH_ROWS=2 BC_CACHE_TRUNK=1 BC_USE_W=1 BC_USE_UW=1 BC_SELF_KL=30 BC_LR=1e-4 ../.venv/bin/python cnet2_train.py $D/cnet2_ds_m3s2$n $D/weights/cnet2_b128m3s2$n.npz 10 '{"C":128,"NB":8,"D":128}' > $D/train_m3s2$n.log 2>&1
done
md5 -q $D/weights/cnet2_b128m3s2{a,b,c}.npz; echo M3S2_TRAINED
