#!/bin/bash
# q9 (dgx): endgame-leaf variants vs the Master2s stand-in (996000, 200; control c_v17d2_m2s2)
cd /home/arm/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=12
B=../.venv/bin/python; P=../data/q9.progress
while pgrep -f "[b]igeval.py exp" >/dev/null; do sleep 15; done
rm -f ../data/bigeval_live/c_v18lr_m2s2.jsonl; $B bigeval.py exp/submission_cnet2_v18lr_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v18lr_m2s2 >> ../data/q9.log 2>&1; echo "$(date +%T) c_v18lr_m2s2" >> $P
rm -f ../data/bigeval_live/c_v18lb100_m2s2.jsonl; $B bigeval.py exp/submission_cnet2_v18lb100_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v18lb100_m2s2 >> ../data/q9.log 2>&1; echo "$(date +%T) c_v18lb100_m2s2" >> $P
echo DONE > ../data/q9.done
