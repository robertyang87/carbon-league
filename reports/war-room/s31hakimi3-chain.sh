#!/bin/bash
# Hakimi v3 calibration double-arm, sequential, full-load recipe (EVO_WORKERS=62).
# Local source; run on spot-31 as /root/carbon/data/hakimi3_chain.sh (setsid-detached).
# Order r1 -> g1; each arm double-field {Master2 996000, Echo2 999000} x 200.
# Packs are self-contained (weights embedded, int8); pre-flight checks pack md5s
# against the war-room-issued values (175fca94 / 597d76f7).
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
[ "$(md5sum exp/hakimi3r1.py | cut -d' ' -f1)" = "175fca94bb173ec9d5f269e9593d2815" ] || { echo "R1 PACK MD5 MISMATCH"; exit 1; }
[ "$(md5sum exp/hakimi3g1.py | cut -d' ' -f1)" = "597d76f77495cb51c1ee56959b5a3e66" ] || { echo "G1 PACK MD5 MISMATCH"; exit 1; }
echo "HAKIMI3_CHAIN_START $(date '+%F %T %Z')" > ../data/hakimi3_chain.log
$PY bigeval.py exp/hakimi3r1.py exp/opp_master2r.py 996000 200 hakimi3r1_m2 > ../data/hakimi3r1_m2.launcher.log 2>&1
echo "R1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/hakimi3_chain.log
$PY bigeval.py exp/hakimi3r1.py exp/echo2.py 999000 200 hakimi3r1_echo2 > ../data/hakimi3r1_echo2.launcher.log 2>&1
echo "R1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/hakimi3_chain.log
$PY bigeval.py exp/hakimi3g1.py exp/opp_master2r.py 996000 200 hakimi3g1_m2 > ../data/hakimi3g1_m2.launcher.log 2>&1
echo "G1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/hakimi3_chain.log
$PY bigeval.py exp/hakimi3g1.py exp/echo2.py 999000 200 hakimi3g1_echo2 > ../data/hakimi3g1_echo2.launcher.log 2>&1
echo "G1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/hakimi3_chain.log
