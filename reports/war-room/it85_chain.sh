#!/bin/bash
# C2 it85 checkpoint calibration, Echo2 field (spot-31). Runs on-machine.
# Field 1: candidate it85 vs Echo2, seed 999000, 200 games.
# Field 2: baseline m3s2r1 (frozen wrapper, weight b6db08c9) vs Echo2, same seeds.
cd /root/carbon/code || exit 9
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "CHAIN_START $(date '+%F %T') pgid=$$"
$PY bigeval.py exp/it85.py exp/echo2.py 999000 200 it85_echo2 > /root/carbon/data/it85_echo2.launcher.log 2> /root/carbon/data/it85_echo2.err.log
echo "FIELD1_RC=$?"
$PY bigeval.py exp/m3s2r1.py exp/echo2.py 999000 200 m3s2r1_echo2_base > /root/carbon/data/m3s2r1_echo2_base.launcher.log 2> /root/carbon/data/m3s2r1_echo2_base.err.log
echo "FIELD2_RC=$?"
echo "IT85_CHAIN_DONE $(date '+%F %T')"
