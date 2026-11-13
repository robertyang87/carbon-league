#!/bin/bash
# DGX: pure-net paired exam of the distilled nets vs Grove1 pure net (same seeds 994000+100, same machine), 3 training opponents.
# usage: dgx_ex1x.sh TAG ARM... (ARM = weights stem in data/weights; "cnet2_b128m2t3" is the control). <= 15 workers total.
cd ~/junyang/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 AWR_US_CONF='{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
G=$1; shift
for W in "$@"; do
  for O in cnet2_m2s2stand cnet2_m2so s1_net; do
    AWR_US_WEIGHTS=$HOME/junyang/carbon/data/weights/$W.npz EVO_WORKERS=5 ../.venv/bin/python bigeval.py awr_us_env.py exp/$O.py 994000 100 ${G}_${W}_$O > ../data/${G}_${W}_$O.out 2>&1 &
  done
  wait
done
echo EXAM_DONE > ../data/${G}_exam.done
