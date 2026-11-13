#!/bin/bash
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
rm -f ../data/bigeval_live/c_v7s8_r10_wb532.jsonl ../data/bigeval_live/c_v7s8_r10_m2ss.jsonl
B=../.venv/bin/python; W=exp/cnet2_v7s_mfxn8_r10.py
ls $W || exit 1
# finish the 10-sim v13 exam: the 20 seeds that were lost when the old queue was stopped
python3 - <<'P'
import json
f='/root/carbon/data/bigeval_live/c_v7s8_r10_v13.jsonl'; R=[json.loads(l) for l in open(f)]
print('have', len(R), 'missing', sorted(set(range(992000,992200))-{r['seed'] for r in R})[:3], '...')
P
EVO_WORKERS=20 $B bigeval.py $W exp/submission_cnet2_v13.py 992180 20 c_v7s8_r10_v13 >> ../data/rsq.log 2>&1; echo "$(date +%T) done r10 v13 tail" >> ../data/rsq.progress
for t in wb532:opp_wb532:995000 m2ss:cnet2_m2sstand:996000; do IFS=: read n f s <<< "$t"
  $B bigeval.py $W exp/$f.py $s 200 c_v7s8_r10_$n >> ../data/rsq.log 2>&1 && echo "$(date +%T) done r10 $n" >> ../data/rsq.progress || echo "$(date +%T) FAILED r10 $n" >> ../data/rsq.progress
done
echo "$(date +%T) done r6 m2ss (all cloud exams finished)" >> ../data/rsq.progress
