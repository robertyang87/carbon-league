#!/bin/bash
# DGX queue (~/junyang): the tail of the cloud q6, taken from the END so the two hosts meet in the middle.
# Same resumable run() as q5/q6; 20 workers (all cores, per the user).
cd ~/junyang/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BC_W=64 EVO_WORKERS=20
B=../.venv/bin/python; P=../data/dq.progress
run() {  # run PACK OPP SEED N TAG
  local f=../data/bigeval_live/$5.jsonl
  if [ -f "$f" ] && [ "$(grep -c '"margin": -\?[0-9]' "$f")" -ge "$4" ]; then echo "$(date +%T) skip $5 (have $4)" >> $P; return; fi
  rm -f "$f"; $B bigeval.py exp/$1.py exp/$2.py $3 $4 $5 >> ../data/dq.log 2>&1; echo "$(date +%T) $5" >> $P
}
run submission_cnet2_v17tb_nc opp_1cz_v9 997000 200 c_v17tb_v9
run submission_cnet2_v18c_nc opp_1cz_v9 997000 200 c_v18c_v9
run submission_cnet2_v17te_nc opp_1cz_v9 997000 200 c_v17te_v9
run submission_cnet2_v17t_nc opp_1cz_v9 997000 200 c_v17t_v9
echo DONE > ../data/dq.done
