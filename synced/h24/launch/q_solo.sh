#!/bin/bash
# solo replicas M6 / K4: int8 no-roll pack (row 2 all game), vs Master2 real and vs E0 (exp/echo_e0.py), 200 each, raw margins
cd /root/carbon/code
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for v in m6r1 k4r1; do ( until [ -f ../data/weights/cnet2_${v}L.npz.ok ]; do sleep 30; done
  CNET2_NO_ROLL=1 CNET2_QMODE=int8 ../.venv/bin/python cnet2_search_pack.py exp/pk_${v}.py "{'SEARCH': False, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}" ../data/weights/cnet2_${v}L.npz > ../data/pk_${v}.log 2>&1
  md5sum exp/pk_${v}.py >> ../data/pk_${v}.log
  for O in opp_master2r echo_e0; do EVO_WORKERS=14 setsid nohup ../.venv/bin/python bigeval.py exp/pk_${v}.py exp/$O.py 994000 200 solo_${v}_$O > ../data/solo_${v}_$O.out 2>&1 < /dev/null & done; wait ) & done
wait; echo DONE > ../data/solo.done
