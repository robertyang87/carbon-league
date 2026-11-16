#!/bin/bash
# S3 enumeration pilot: 2 lanes concurrently, 1 game each (games 3 and 4).
# Each lane runs in its own systemd scope (CPUQuota 150%, MemoryMax 512M),
# BLAS pinned to 1, independent output dir. Run on spot-34, cwd=/root/carbon.
set -u
cd /root/carbon || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/root/carbon/.venv/bin/python
SCRIPT=/root/carbon/s3_enum/s3_freeze/enumerate_roots.py
PACK=/tmp/echo4_int8.py
date +%s > /root/carbon/s3_enum/s3_pilot_start.txt
systemd-run --scope -p CPUQuota=150% -p MemoryMax=512M --unit=s3enum-c1-l1 \
  /bin/bash -c "exec > /root/carbon/s3_enum/lane1.log 2>&1; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $PY $SCRIPT --games 3-3 --out /root/carbon/s3_enum/out_c1_lane1 --pack $PACK --shard-id c1_l1"
systemd-run --scope -p CPUQuota=150% -p MemoryMax=512M --unit=s3enum-c1-l2 \
  /bin/bash -c "exec > /root/carbon/s3_enum/lane2.log 2>&1; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $PY $SCRIPT --games 4-4 --out /root/carbon/s3_enum/out_c1_lane2 --pack $PACK --shard-id c1_l2"
sleep 5
echo "scopes:"; systemctl list-units --type=scope --all | grep s3enum || true
echo "procs:"; pgrep -af "enumerate_root[s]" || echo 0
date
