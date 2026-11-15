#!/bin/bash
echo "N1RT6F3Z_BEGIN"
echo "== baseline field BEFORE =="
wc -l /root/carbon/data/bigeval_live/e0_m2_base.jsonl
ps -p 119669 -o pid,etime,args --no-headers | head -1
echo "== pack md5 =="
md5sum /root/carbon/data/packs/n1r_int8.py
# stage the timing script (sent along as base64 in N1RT_PY_B64 env-substitute below)
echo "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uMwoiIiJuMXIgbGFkZGVyLXRpbWluZyBwcm9iZTogT05FIGdhbWUgdnMgZWNob19lMCwgcmVwbGljYXRlZCBmcm9tCmV2YWxfdjUzLnBsYXlfZ2FtZSdzIGxvb3AgKHNhbWUgc2VlZGluZywgc2FtZSBGYXN0Q2FyYm9uLCBzYW1lIENGRyksCnJlY29yZGluZyB3YWxsIHRpbWUgb2YgZXZlcnkgbjFyIGFnZW50KCkgY2FsbC4gSW4tcHJvY2VzcyB0aW1pbmcgb25seS4iIiIKaW1wb3J0IGpzb24sIG9zLCByYW5kb20sIHN0YXRpc3RpY3MgYXMgc3QsIHN5cywgdGltZQoKQ09ERSA9ICIvcm9vdC9jYXJib24vY29kZSIKc3lzLnBhdGguaW5zZXJ0KDAsIENPREUpCm9zLmNoZGlyKENPREUpCgpTRUVEID0gNzAwMDAxCkFfU0lERSA9IFNFRUQgJSAyICAjIGJpZ2V2YWwgY29udmVudGlvbjogc2VhdCBhbHRlcm5hdGVzIGJ5IHNlZWQgcGFyaXR5Cgp0X3N0YXJ0ID0gdGltZS5wZXJmX2NvdW50ZXIoKQppbXBvcnQgZXZhbF92NTMgICMgbm9xYTogRTQwMiAgKHByb3ZpZGVzIENGRywgRmFzdENhcmJvbiwgbG9hZF9hZ2VudF9mbiwgcm5nX3NlZWQsIE5TVEVQUykKQ0ZHID0gZXZhbF92NTMuQ0ZHCgpyZXMgPSB7InNlZWQiOiBTRUVELCAiYV9zaWRlIjogQV9TSURFLCAibnN0ZXBzX2VudiI6IGV2YWxfdjUzLk5TVEVQU30KCiMgLS0tIGluaXQgcGFydCAxOiBtb2R1bGUgaW1wb3J0IChsb2FkX2FnZW50X2ZuIGV4ZWNzIHRoZSBwYWNrIG1vZHVsZSkgLS0tCnQwID0gdGltZS5wZXJmX2NvdW50ZXIoKQpmYSA9IGV2YWxfdjUzLmxvYWRfYWdlbnRfZm4ob3MucGF0aC5qb2luKENPREUsICIuLiIsICJkYXRhIiwgInBhY2tzIiwgIm4xcl9pbnQ4LnB5IikpCnJlc1siaW1wb3J0X21zIl0gPSByb3VuZCgodGltZS5wZXJmX2NvdW50ZXIoKSAtIHQwKSAqIDEwMDAsIDEpCmZiID0gZXZhbF92NTMubG9hZF9hZ2VudF9mbihvcy5wYXRoLmpvaW4oQ09ERSwgImV4cCIsICJlY2hvX2UwLnB5IikpCgojIC0tLSBnYW1lLCBleGFjdCBwbGF5X2dhbWUgcmVwbGljYXRpb24gKG1pbnVzIGxlZGdlcnMsIHdoaWNoIHRvdWNoIG5laXRoZXIgc2ltIG5vciBhZ2VudHMpIC0tLQpyYW5kb20uc2VlZChldmFsX3Y1My5ybmdfc2VlZChTRUVELCBBX1NJREUpKQpzaW0gPSBldmFsX3Y1My5GYXN0Q2FyYm9uKENGRykucmVzZXQoc2VlZD1TRUVEKQphZ2VudHMgPSBbTm9uZSwgTm9uZV0KYWdlbnRzW0FfU0lERV0gPSBmYQphZ2VudHNbMSAtIEFfU0lERV0gPSBmYgoKc3RlcF9tcyA9IFtdCmVyciA9IE5vbmUKdF9nYW1lMCA9IHRpbWUucGVyZl9jb3VudGVyKCkKdHJ5OgogICAgZm9yIHN0ZXAgaW4gcmFuZ2UoMSwgZXZhbF92NTMuTlNURVBTICsgMSk6CiAgICAgICAgYWN0cyA9IFtdCiAgICAgICAgZm9yIHAgaW4gcmFuZ2UoMik6CiAgICAgICAgICAgIG8gPSBzaW0ub2JzKHApCiAgICAgICAgICAgIHRzID0gdGltZS5wZXJmX2NvdW50ZXIoKQogICAgICAgICAgICBhID0gYWdlbnRzW3BdKG8sIENGRykKICAgICAgICAgICAgZHQgPSAodGltZS5wZXJmX2NvdW50ZXIoKSAtIHRzKSAqIDEwMDAKICAgICAgICAgICAgYWN0cy5hcHBlbmQoYSkKICAgICAgICAgICAgaWYgcCA9PSBBX1NJREU6CiAgICAgICAgICAgICAgICBzdGVwX21zLmFwcGVuZChkdCkKICAgICAgICBzaW0uc3RlcChhY3RzWzBdLCBhY3RzWzFdKQpleGNlcHQgRXhjZXB0aW9uIGFzIGU6CiAgICBlcnIgPSByZXByKGUpWzozMDBdCnJlc1siZ2FtZV93YWxsX3MiXSA9IHJvdW5kKHRpbWUucGVyZl9jb3VudGVyKCkgLSB0X2dhbWUwLCAyKQpyZXNbImVycm9yIl0gPSBlcnIKCmlmIHN0ZXBfbXM6CiAgICBzcnQgPSBzb3J0ZWQoc3RlcF9tcykKICAgIG4gPSBsZW4oc3J0KQogICAgcDk1ID0gc3J0W21pbihuIC0gMSwgaW50KDAuOTUgKiBuKSldCiAgICByZXMudXBkYXRlKHsKICAgICAgICAibl9zdGVwcyI6IG4sCiAgICAgICAgImZpcnN0X2NhbGxfbXMiOiByb3VuZChzdGVwX21zWzBdLCAxKSwgICAgICAgICAgIyBpbmNsdWRlcyBsYXp5IGRlcXVhbnRpemUgKyBhZ2VudCBjb25zdHJ1Y3Rpb24gKyAxc3QgZGVjaXNpb24KICAgICAgICAiaW5pdF90b3RhbF9tcyI6IHJvdW5kKHJlc1siaW1wb3J0X21zIl0gKyBzdGVwX21zWzBdLCAxKSwKICAgICAgICAicDUwX21zIjogcm91bmQoc3QubWVkaWFuKHN0ZXBfbXMpLCAxKSwKICAgICAgICAicDk1X21zIjogcm91bmQocDk1LCAxKSwKICAgICAgICAibWF4X21zIjogcm91bmQoc3J0Wy0xXSwgMSksCiAgICAgICAgInNsb3dfc3RlcHNfZ3QxMDAwIjogc3VtKDEgZm9yIHggaW4gc3RlcF9tcyBpZiB4ID4gMTAwMCksCiAgICAgICAgImV4dHJhX21zIjogcm91bmQoc3VtKG1heCgwLjAsIHggLSAxMDAwLjApIGZvciB4IGluIHN0ZXBfbXMpLCAxKSwKICAgICAgICAic3VtX3N0ZXBzX3MiOiByb3VuZChzdW0oc3RlcF9tcykgLyAxMDAwLCAyKSwKICAgIH0pCnRyeToKICAgIHdpdGggb3BlbigiL3Byb2Mvc2VsZi9zdGF0dXMiKSBhcyBmOgogICAgICAgIGZvciBsbiBpbiBmOgogICAgICAgICAgICBpZiBsbi5zdGFydHN3aXRoKCJWbUhXTSIpOgogICAgICAgICAgICAgICAgcmVzWyJwZWFrX3Jzc19rYiJdID0gaW50KGxuLnNwbGl0KClbMV0pCmV4Y2VwdCBFeGNlcHRpb246CiAgICBwYXNzCnByaW50KCJOMVJUSU1FX0pTT04gIiArIGpzb24uZHVtcHMocmVzKSkK" | base64 -d > /root/carbon/code/n1r_timing_probe.py
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
