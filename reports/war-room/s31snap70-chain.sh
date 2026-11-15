#!/bin/bash
# snap_70 calibration, Echo2 field only, full-load recipe (EVO_WORKERS=62).
# Local source; run on spot-31 as /root/carbon/data/snap70_chain.sh (setsid-detached).
# Field: vs Echo2 999000 x 200, bare-net pack. C1 runs the Master2 field in parallel on spot-29.
# Pack is self-contained (weights embedded, int8); pre-flight checks pack md5
# against the war-room-issued value (ece3ff52).
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
[ "$(md5sum exp/snap_70.py | cut -d' ' -f1)" = "ece3ff5294aaca0be0563be66571d9d7" ] || { echo "SNAP70 PACK MD5 MISMATCH"; exit 1; }
echo "SNAP70_CHAIN_START $(date '+%F %T %Z')" > ../data/snap70_chain.log
$PY bigeval.py exp/snap_70.py exp/echo2.py 999000 200 snap_70_echo2 > ../data/snap_70_echo2.launcher.log 2>&1
echo "SNAP70_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/snap70_chain.log
