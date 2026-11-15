#!/bin/bash
# C2 it95 smoke (spot-31, on-machine): candidate it95 vs Echo2, seed 999000, 2 games.
cd /root/carbon/code || exit 9
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
echo "SMOKE_START $(date '+%F %T') pgid=$$"
$PY bigeval.py exp/it95.py exp/echo2.py 999000 2 it95_smoke > /root/carbon/data/it95_smoke.launcher.log 2> /root/carbon/data/it95_smoke.err.log
echo "SMOKE_RC=$?"
echo "IT95_SMOKE_DONE $(date '+%F %T')"
