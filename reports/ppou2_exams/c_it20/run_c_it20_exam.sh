#!/bin/bash
# c it=20 checkpoint exam (strategy-group ruling 2026-11-15 17:37; registered
# control-arm exam): M2 field then E2 field, sequential, pinned env.
# Subject pack ppou2_c_it20.py md5 7a9a3e974e6f064009cd9527683abbea
# (ckpt_20.npz f043b1f0ef9dc73b871f96d46fb9a9d7, from the spot mirror).
# NOTE: runs while m trains on the same machine; it=27 pace may be polluted
# by exam CPU load -- do not read it=27 gpm as a clean pace sample.
cd /root/carbon/probe_m2/code || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 EVO_WORKERS=24
PY=/root/carbon/.venv/bin/python
L=/root/carbon/probe_m2/c_it20_exam.log
echo "CHAIN_START $(date '+%F %T %Z')" >> $L
echo "M2_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/ppou2_c_it20.py exp/opp_master2r.py 980000 2000 ppou2_c_it20_m2 >> $L 2>&1
echo "M2_RC=$? $(date '+%F %T %Z')" >> $L
echo "E2_START $(date '+%F %T %Z')" >> $L
$PY bigeval.py exp/ppou2_c_it20.py exp/echo2.py 982000 2000 ppou2_c_it20_e2 >> $L 2>&1
echo "E2_RC=$? $(date '+%F %T %Z')" >> $L
echo DONE > /root/carbon/probe_m2/c_it20_exam_done.txt
