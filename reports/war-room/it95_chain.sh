#!/bin/bash
# C2 it95 checkpoint exam, Echo2 field (spot-31, on-machine chain).
# Field 1: candidate it95 (snipe_n1 it95 frozen net) vs Echo2, seed 999000, 200 games.
# Field 2: baseline m3s2r1 (pre-U frozen wrapper) vs Echo2, same seeds, paired.
cd /root/carbon/code || exit 9
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "ENV_SEGMENT: EVO_WORKERS=$EVO_WORKERS OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "CMD1: $PY bigeval.py exp/it95.py exp/echo2.py 999000 200 it95_echo2"
echo "CMD2: $PY bigeval.py exp/m3s2r1.py exp/echo2.py 999000 200 m3s2r1_echo2_base95"
echo "CHAIN_START $(date '+%F %T') pgid=$$"
M=$(md5sum exp/it95.py | cut -d' ' -f1)
echo "PACK_MD5 exp/it95.py=$M"
[ "$M" = "9e16085acd3549df210c6209ab3fd3cb" ] || { echo PACK_MD5_MISMATCH; exit 8; }
$PY bigeval.py exp/it95.py exp/echo2.py 999000 200 it95_echo2 > /root/carbon/data/it95_echo2.launcher.log 2> /root/carbon/data/it95_echo2.err.log
echo "FIELD1_RC=$?"
$PY bigeval.py exp/m3s2r1.py exp/echo2.py 999000 200 m3s2r1_echo2_base95 > /root/carbon/data/m3s2r1_echo2_base95.launcher.log 2> /root/carbon/data/m3s2r1_echo2_base95.err.log
echo "FIELD2_RC=$?"
echo "IT95_CHAIN_DONE $(date '+%F %T')"
