#!/bin/bash
# v17 search + dg2 policy, paired with c_v17c_* on the same maps; runs next to te.sh (30 workers each way)
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64 EVO_WORKERS=30
while [ ! -f ../data/night.done ]; do sleep 30; done
B=../.venv/bin/python
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/cnet2_m2s2stand.py 996000 200 c_v17d2_m2s2 >> ../data/te0.log 2>&1; echo "$(date +%T) v17d2 m2s2" >> ../data/te.progress
$B bigeval.py exp/submission_cnet2_v17d2_nc.py exp/opp_1cz_v9.py 997000 200 c_v17d2_v9 >> ../data/te0.log 2>&1; echo "$(date +%T) v17d2 v9" >> ../data/te.progress
echo DONE > ../data/te0.done
