#!/bin/bash
# it320 calibration triple-arm, sequential, full-load recipe (EVO_WORKERS=62).
# Local source; run on spot-31 as /root/carbon/data/it320_chain.sh (setsid-detached).
# Order r1 -> g1 -> wr1; each arm double-field {Master2 996000, Echo2 999000} x 200.
# Packs are self-contained (weights embedded, int8); pre-flight checks pack md5s
# against the engineering-issued values (ce4d4fdb / e826f2b0 / fe75eade).
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
[ "$(md5sum exp/m9xj320r1.py | cut -d' ' -f1)" = "ce4d4fdbf0aef3025ac85a65652400d6" ] || { echo "R1 PACK MD5 MISMATCH"; exit 1; }
[ "$(md5sum exp/m9xj320g1.py | cut -d' ' -f1)" = "e826f2b0271b15a5fcc7ccc218e8fe85" ] || { echo "G1 PACK MD5 MISMATCH"; exit 1; }
[ "$(md5sum exp/m9xj320wr1.py | cut -d' ' -f1)" = "fe75eade96b9f50291d9db6cf0baf43f" ] || { echo "WR1 PACK MD5 MISMATCH"; exit 1; }
echo "IT320_CHAIN_START $(date '+%F %T %Z')" > ../data/it320_chain.log
$PY bigeval.py exp/m9xj320r1.py exp/opp_master2r.py 996000 200 m9xj320r1_m2 > ../data/m9xj320r1_m2.launcher.log 2>&1
echo "R1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
$PY bigeval.py exp/m9xj320r1.py exp/echo2.py 999000 200 m9xj320r1_echo2 > ../data/m9xj320r1_echo2.launcher.log 2>&1
echo "R1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
$PY bigeval.py exp/m9xj320g1.py exp/opp_master2r.py 996000 200 m9xj320g1_m2 > ../data/m9xj320g1_m2.launcher.log 2>&1
echo "G1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
$PY bigeval.py exp/m9xj320g1.py exp/echo2.py 999000 200 m9xj320g1_echo2 > ../data/m9xj320g1_echo2.launcher.log 2>&1
echo "G1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
$PY bigeval.py exp/m9xj320wr1.py exp/opp_master2r.py 996000 200 m9xj320wr1_m2 > ../data/m9xj320wr1_m2.launcher.log 2>&1
echo "WR1_M2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
$PY bigeval.py exp/m9xj320wr1.py exp/echo2.py 999000 200 m9xj320wr1_echo2 > ../data/m9xj320wr1_echo2.launcher.log 2>&1
echo "WR1_ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/it320_chain.log
