#!/bin/bash
# Opening exam (stop at step 100), pure net, Mac. usage: open100.sh NAME OPP SEED0 N   (NAME = exp/cnet2_NAME.py; OPP m2s2|v9)
# tag o100_OPP_NAME. 400 games on one seed range = both former map sets in one paired sample.
cd /Users/baggioustb/dev/carbon/carbon-league/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=4 EVAL_STEPS=100
case $2 in m2s2) O=exp/cnet2_m2s2stand.py;; v9) O=exp/opp_1cz_v9.py;; m2so) O=exp/cnet2_m2so.py;; m2r) O=exp/ropen_m2r.py;; *) echo bad opp; exit 1;; esac
T=o100_$2_$1; rm -f ../data/bigeval_live/$T.jsonl
../.venv/bin/python bigeval.py exp/cnet2_$1.py $O $3 $4 $T >> ../data/open100.log 2>&1 < /dev/null
echo "$(date +%T) $T $(wc -l < ../data/bigeval_live/$T.jsonl)" >> ../data/open100.progress
