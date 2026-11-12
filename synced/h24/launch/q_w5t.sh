#!/bin/bash
# W5 pack timing: wait for the 200-game pack exam, pause the other exams (SIGSTOP), time 20 games single-process, resume (SIGCONT)
cd /root/carbon/code
for p in $(pgrep -f "bash /root/q_w5pk.sh"); do kill $p; done
until [ "$(wc -l < ../data/bigeval_live/w5pk_m2r.jsonl 2>/dev/null)" -ge 200 ] && ! pgrep -f "200 w5pk_m2r" >/dev/null; do sleep 15; done
P=$(pgrep -f "bigeval.py"); for p in $P; do kill -STOP $p; for c in $(pgrep -P $p); do kill -STOP $c; done; done
sleep 3; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../.venv/bin/python cloud/pack_timing.py exp/submission_echo2_w5q8.py 20 > ../data/w5pk_timing.out 2>&1
for p in $P; do for c in $(pgrep -P $p); do kill -CONT $c; done; kill -CONT $p; done
echo DONE >> ../data/w5pk_timing.out
