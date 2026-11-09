#!/bin/bash
# Opening exam on a host: stop at step 100.  usage: o100.sh CODE_DIR WORKERS PACK OPP SEED0 N TAG [PACK OPP SEED0 N TAG ...]
cd "$1" || exit 1; W=$2; shift 2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$W EVAL_STEPS=100
while [ $# -ge 5 ]; do
  rm -f ../data/bigeval_live/$5.jsonl
  ../.venv/bin/python bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/o100.log 2>&1 < /dev/null
  echo "$(date +%T) $5 $(wc -l < ../data/bigeval_live/$5.jsonl)" >> ../data/o100.progress; shift 5
done
echo DONE >> ../data/o100.progress
