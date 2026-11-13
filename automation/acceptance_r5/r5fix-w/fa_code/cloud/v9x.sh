#!/bin/bash
# true-foreign exam: 1C-Zero-v9 public source (approved local exam). Search vs no-search on the same maps.
cd ~/carbon/code; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BC_W=64
while [ ! -f ../data/v16pk.done ]; do sleep 20; done
EVO_WORKERS=60 ../.venv/bin/python bigeval.py exp/cnet2_v7s57_240.py exp/opp_1cz_v9.py 997000 200 c_v16n_v9 >> ../data/v9x.log 2>&1
EVO_WORKERS=60 ../.venv/bin/python bigeval.py exp/submission_cnet2_v16s.py exp/opp_1cz_v9.py 997000 200 c_v16spk_v9 >> ../data/v9x.log 2>&1
echo DONE > ../data/v9x.done
