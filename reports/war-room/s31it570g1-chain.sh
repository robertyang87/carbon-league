#!/bin/bash
# it570 g1 calibration double-field, sequential, full-load recipe (EVO_WORKERS=62).
# Local source; run on spot-31 as /root/carbon/data/it570g1_chain.sh (setsid-detached).
# Fields: {Master2 996000, Echo2 999000} x 200, bare-net pack.
# Pack is self-contained (weights embedded, int8); pre-flight checks pack md5
# against the war-room-issued value (e6e190a8).
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
[ "$(md5sum exp/m9xj570g1.py | cut -d' ' -f1)" = "e6e190a8f16250b1ffa45d5913d42d97" ] || { echo "G1 PACK MD5 MISMATCH"; exit 1; }
echo "IT570G1_CHAIN_START $(date '+%F %T %Z')" > ../data/it570g1_chain.log
$PY bigeval.py exp/m9xj570g1.py exp/opp_master2r.py 996000 200 m9xj570g1_m2 > ../data/m9xj570g1_m2.launcher.log 2>&1
echo "G1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it570g1_chain.log
$PY bigeval.py exp/m9xj570g1.py exp/echo2.py 999000 200 m9xj570g1_echo2 > ../data/m9xj570g1_echo2.launcher.log 2>&1
echo "G1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it570g1_chain.log
