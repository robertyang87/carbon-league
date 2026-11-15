#!/bin/bash
echo "N1RT6F3Z_BEGIN"
echo "== baseline field BEFORE =="
wc -l /root/carbon/data/bigeval_live/e0_m2_base.jsonl
ps -p 119669 -o pid,etime,args --no-headers | head -1
echo "== pack md5 =="
md5sum /root/carbon/data/packs/n1r_int8.py
# stage the timing script (sent along as base64 in N1RT_PY_B64 env-substitute below)
echo "$N1RT_PY_B64" | base64 -d > /root/carbon/code/n1r_timing_probe.py
wc -c /root/carbon/code/n1r_timing_probe.py

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
PY=/root/carbon/.venv/bin/python
CG=/sys/fs/cgroup/n1rtime_probe
MECH=""

# --- try cgroup v2 first ---
( echo +cpu +memory > /sys/fs/cgroup/cgroup.subtree_control ) 2>/dev/null
mkdir -p "$CG" 2>/dev/null
if echo "150000 100000" > "$CG/cpu.max" 2>/dev/null && echo 536870912 > "$CG/memory.max" 2>/dev/null; then
  if [ "$(cat $CG/cpu.max)" = "150000 100000" ] && [ "$(cat $CG/memory.max)" = "536870912" ]; then
    MECH="cgroup-v2"
  fi
fi

cd /root/carbon/code
T0=$(date +%s)
if [ "$MECH" = "cgroup-v2" ]; then
  echo "MECH=cgroup-v2 cpu.max=$(cat $CG/cpu.max) memory.max=$(cat $CG/memory.max)"
  nice -n 19 setsid bash -c "echo \$\$ > $CG/cgroup.procs; exec $PY /root/carbon/code/n1r_timing_probe.py" 2>&1
  RC=$?
  echo "cgroup memory.peak=$(cat $CG/memory.peak 2>/dev/null) cpu.stat:"; grep -E "usage_usec|nr_throttled|throttled_usec" $CG/cpu.stat 2>/dev/null
  # cleanup: move nothing (process exited), remove group
  rmdir "$CG" 2>/dev/null
elif command -v systemd-run >/dev/null; then
  MECH="systemd-run"
  echo "MECH=systemd-run CPUQuota=150% MemoryMax=512M"
  nice -n 19 systemd-run --scope --quiet -p CPUQuota=150% -p MemoryMax=512M $PY /root/carbon/code/n1r_timing_probe.py 2>&1
  RC=$?
else
  MECH="taskset-fallback"
  echo "MECH=taskset 2 cores + OMP_NUM_THREADS=2"
  nice -n 19 taskset -c 0,1 $PY /root/carbon/code/n1r_timing_probe.py 2>&1
  RC=$?
fi
T1=$(date +%s)
echo "probe_rc=$RC wall_outer_s=$((T1-T0)) mech=$MECH"
echo "== baseline field AFTER =="
wc -l /root/carbon/data/bigeval_live/e0_m2_base.jsonl
ps -p 119669 -o pid,etime,args --no-headers | head -1
echo "N1RT6F3Z_END"
