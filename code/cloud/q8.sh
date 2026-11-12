#!/bin/bash
# q8: v17d2 and v17 vs 1C-v9 on a SECOND map set (seeds 997200-997399) to pin down v17d2's real level vs 1C.
# Chunks of 50 games, the two versions interleaved, so a preemption (dm2 search exams) keeps every finished chunk paired;
# rerunning q8 skips finished chunks.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=$(( $(nproc) - 4 ))
B=../.venv/bin/python; P=../data/q8.progress
run() {  # run PACK OPP SEED N TAG
  local f=../data/bigeval_live/$5.jsonl
  if [ -f "$f" ] && [ "$(grep -c '"margin": -\?[0-9]' "$f")" -ge "$4" ]; then echo "$(date +%T) skip $5" >> $P; return; fi
  rm -f "$f"; $B bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/q8.log 2>&1; echo "$(date +%T) $5" >> $P
}
while pgrep -f "[b]igeval.py exp" >/dev/null; do sleep 15; done
for c in 0 1 2 3; do s=$((997200 + 50 * c))
  run submission_cnet2_v17d2_nc opp_1cz_v9 $s 50 c_v17d2_v9b_p$c
  run submission_cnet2_v17c_nc opp_1cz_v9 $s 50 c_v17c_v9b_p$c
done
echo DONE > ../data/q8.done
