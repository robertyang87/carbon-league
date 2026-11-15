#!/bin/bash
# C1 n1r vs Echo3 main field (spot-29, on-machine): A=n1r int8 pack (ours), B=Echo3 int8 pack, seeds 989000-989199, 200 games, deployment form as-is.
cd /root/carbon/code || exit 9
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "ENV_SEGMENT: EVO_WORKERS=$EVO_WORKERS OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "CMD: $PY bigeval.py exp/n1r_int8.py exp/echo3_int8.py 989000 200 n1r_echo3"
echo "MAIN_START $(date '+%F %T') pgid=$$"
M1=$(md5sum exp/n1r_int8.py | cut -d' ' -f1)
echo "PACK_MD5 exp/n1r_int8.py=$M1"
[ "$M1" = "ba7f84c3652ea2cd14a2243f2b1eace2" ] || { echo PACK_MD5_MISMATCH_N1R; exit 8; }
M2=$(md5sum exp/echo3_int8.py | cut -d' ' -f1)
echo "PACK_MD5 exp/echo3_int8.py=$M2"
[ "$M2" = "7b7620499411a088cf4dc17c0e929bd7" ] || { echo PACK_MD5_MISMATCH_ECHO3; exit 8; }
$PY bigeval.py exp/n1r_int8.py exp/echo3_int8.py 989000 200 n1r_echo3 > /root/carbon/data/n1r_echo3.launcher.log 2> /root/carbon/data/n1r_echo3.err.log
echo "MAIN_RC=$?"
echo "N1RE3_MAIN_DONE $(date '+%F %T')"
