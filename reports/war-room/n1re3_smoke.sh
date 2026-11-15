#!/bin/bash
# C1 n1r vs Echo3 smoke (spot-29, on-machine): A=n1r int8 pack, B=Echo3 int8 pack, seed 989000, 2 games.
cd /root/carbon/code || exit 9
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "SMOKE_START $(date '+%F %T') pgid=$$"
M1=$(md5sum exp/n1r_int8.py | cut -d' ' -f1)
echo "PACK_MD5 exp/n1r_int8.py=$M1"
[ "$M1" = "ba7f84c3652ea2cd14a2243f2b1eace2" ] || { echo PACK_MD5_MISMATCH_N1R; exit 8; }
M2=$(md5sum exp/echo3_int8.py | cut -d' ' -f1)
echo "PACK_MD5 exp/echo3_int8.py=$M2"
[ "$M2" = "7b7620499411a088cf4dc17c0e929bd7" ] || { echo PACK_MD5_MISMATCH_ECHO3; exit 8; }
$PY bigeval.py exp/n1r_int8.py exp/echo3_int8.py 989000 2 n1r_echo3_smoke > /root/carbon/data/n1r_echo3_smoke.launcher.log 2> /root/carbon/data/n1r_echo3_smoke.err.log
echo "SMOKE_RC=$?"
echo "N1RE3_SMOKE_DONE $(date '+%F %T')"
