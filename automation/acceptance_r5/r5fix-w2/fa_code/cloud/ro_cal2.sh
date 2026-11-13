#!/bin/bash
# calibrate the rule M stand-in against dg5 (100 steps, 100 games). usage: ro_cal.sh CODE_DIR WORKERS PART NPARTS
# (each host takes lines with index % NPARTS == PART; finished tags are skipped)
cd "$1"; W=$2; PART=$3; NP=$4; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$W EVAL_STEPS=100
i=0; while read n cash; do i=$((i+1)); [ $((i % NP)) -eq $PART ] || continue
  f=../data/bigeval_live/ro_$n.jsonl; [ -f $f ] && [ $(wc -l < $f) -ge 100 ] && continue
  rm -f $f; EVAL_OPP_CASH=$cash ../.venv/bin/python bigeval.py exp/cnet2_dg5e5s57.py exp/ropen_$n.py 996000 100 ro_$n > /dev/null 2>&1 < /dev/null
  echo "$(date +%T) $n" >> ../data/ro_cal.progress
done < cloud/ro_cal2.txt
echo "PART_DONE $PART" >> ../data/ro_cal.progress
