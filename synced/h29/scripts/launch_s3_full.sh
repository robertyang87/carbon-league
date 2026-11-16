#!/bin/bash
# S3 enumeration full expansion: 16 lanes, games 5-269 (265 games).
# 9 lanes x 17 + 7 lanes x 16 = 265; contiguous, non-overlapping, union = 5-269.
# Each lane: own systemd scope (CPUQuota 150%, MemoryMax 512M), BLAS pinned 1,
# own output dir. Run on spot-34, cwd=/root/carbon.
# NOTE: systemd-run --scope blocks its caller on this host, so each invocation
# is backgrounded (&) with a short sleep; scopes survive under systemd.
set -u
cd /root/carbon || exit 1
PY=/root/carbon/.venv/bin/python
SCRIPT=/root/carbon/s3_enum/s3_freeze/enumerate_roots.py
PACK=/tmp/echo4_int8.py
date +%s > /root/carbon/s3_enum/s3_full_start.txt
launch() {
  local lane=$1 g0=$2 g1=$3
  systemd-run --scope --no-block -p CPUQuota=150% -p MemoryMax=512M --unit="s3enum-c1-l${lane}" \
    /bin/bash -c "exec > /root/carbon/s3_enum/lane${lane}.log 2>&1; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 $PY $SCRIPT --games ${g0}-${g1} --out /root/carbon/s3_enum/out_c1_lane${lane} --pack $PACK --shard-id c1_l${lane}" \
    > "/root/carbon/s3_enum/run-l${lane}.log" 2>&1 &
  sleep 2
}
launch 3 5 21
launch 4 22 38
launch 5 39 55
launch 6 56 72
launch 7 73 89
launch 8 90 106
launch 9 107 123
launch 10 124 140
launch 11 141 157
launch 12 158 173
launch 13 174 189
launch 14 190 205
launch 15 206 221
launch 16 222 237
launch 17 238 253
launch 18 254 269
sleep 10
echo "launched 16 lanes; scopes:"; systemctl list-units --type=scope --all 2>/dev/null | grep -c s3enum || true
pgrep -c -f "enumerate_root[s]" || echo 0
date
