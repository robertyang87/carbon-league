#!/bin/bash
# v18 = search-layer configs on the v17 net (b128v9): endgame 3 keys x 4 sims from step 280, ENDGAME_STOP 2;
# v18b adds the Master-line early switch (LATE_STEP 220 when the opponent's first recruit is a collector).
# Exam: Master2s stand-in v2, paired vs v17c on the same maps; key metric = cargo we seize in the last 15 steps.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=60
while [ ! -f ../data/plbase.done ]; do sleep 20; done
for v in v18a v18b; do ../.venv/bin/python bigeval.py exp/submission_cnet2_${v}_nc.py exp/cnet2_m2s2stand.py 996000 200 c_${v}_m2s2 >> ../data/v18.log 2>&1; done
echo DONE > ../data/v18.done
