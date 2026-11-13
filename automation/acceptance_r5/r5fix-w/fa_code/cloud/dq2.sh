#!/bin/bash
# DGX after dg5 training: 8 exam workers next to the dg5 probes (cap 80% = 16 cores). Takes the job host 4 reaches
# last in q6c (v17d2 vs the 无始 stand-in), so sync_hosts can hand it over before host 4 gets there.
cd ~/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=8
until grep -q TRAIN_DONE ../data/cnet2_train_b128dg5.log; do sleep 30; done
B=../.venv/bin/python; P=../data/dq.progress
f=../data/bigeval_live/c_v17d2_ws.jsonl
if [ ! -f "$f" ]; then $B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/cnet2_wsstand.py 999500 100 c_v17d2_ws >> ../data/dq.log 2>&1; echo "$(date +%T) c_v17d2_ws" >> $P; fi
echo DONE > ../data/dq2.done
