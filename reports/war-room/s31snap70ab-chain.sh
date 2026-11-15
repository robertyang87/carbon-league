#!/bin/bash
# snap_70 fastsim A/B: new fastsim (aefe6c83) vs old record (121/200 wins, +237.0). Echo2 field only.
# Runs on spot-31 as /root/carbon/data/snap70ab_chain.sh (setsid-detached).
# Field: vs Echo2 999000 x 200, bare-net pack. Measurement only, NOT a re-exam; no verdict change.
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
[ "$(md5sum exp/snap_70.py | cut -d' ' -f1)" = "ece3ff5294aaca0be0563be66571d9d7" ] || { echo "SNAP70 PACK MD5 MISMATCH"; exit 1; }
[ "$(md5sum fastsim.py | cut -d' ' -f1)" = "aefe6c83826efae4de83bef8c643c80b" ] || { echo "FASTSIM MD5 MISMATCH"; exit 1; }
echo "SNAP70AB_CHAIN_START $(date '+%F %T %Z')" > ../data/snap70ab_chain.log
$PY bigeval.py exp/snap_70.py exp/echo2.py 999000 200 snap_70_echo2_newsim > ../data/snap_70_echo2_newsim.launcher.log 2>&1
echo "SNAP70AB_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/snap70ab_chain.log
