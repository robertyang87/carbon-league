#!/bin/bash
# k4v2r1 field 2 only (vs Echo2, seed 999000) on spot-31, full-load recipe (EVO_WORKERS=62).
# Local source; run on the machine as /root/carbon/data/k4v2r1_field2.sh (nohup, after EXAM-READY + on-machine four-item verification).
# Field 1 (vs Master2) is DONE on spot-30; do not re-run it here.
cd /root/carbon/code || exit 1
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
W=/root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz
# Pre-flight: weight + wrapper must match the spot-30 verified values.
echo "$W" | grep -q . || exit 1
[ "$(md5sum "$W" | cut -d' ' -f1)" = "344cfde939debb0de7f8834af069ea16" ] || { echo "WEIGHT MD5 MISMATCH"; exit 1; }
[ "$(md5sum exp/k4v2r1.py | cut -d' ' -f1)" = "c395faa8f3d783c509cb2f9fc82db676" ] || { echo "WRAPPER MD5 MISMATCH"; exit 1; }
echo "FIELD2_START $(date '+%F %T %Z')" > ../data/k4v2r1_field2.log
$PY bigeval.py exp/k4v2r1.py exp/echo2.py 999000 200 k4v2r1_echo2 > ../data/k4v2r1_echo2.launcher.log 2>&1
echo "ECHO2_DONE rc=$? $(date '+%F %T %Z')" >> ../data/k4v2r1_field2.log
