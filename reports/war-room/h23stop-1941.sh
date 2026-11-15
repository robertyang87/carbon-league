#!/bin/bash
# h23: snapshot m3s2g1_m2 exam procs, STOP the exam tree (owner order), verify, then read-only checks.
MARK="H23STOP1941M3"
TAG="m3s2g1_m2"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
hostname
echo "--- CPU/MEM (pre-stop) ---"
echo "nproc: $(nproc)"
grep -m1 'model name' /proc/cpuinfo
echo "processor entries: $(grep -c '^processor' /proc/cpuinfo)"
free -m
cat /proc/loadavg
echo "--- EXAM PROC SNAPSHOT (comm=python, cmdline contains $TAG) ---"
PIDS=""
MAINPID=""
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  comm=$(cat $d/comm 2>/dev/null) || continue
  case "$comm" in *python*) ;; *) continue;; esac
  cmd=$(tr '\0' ' ' < $d/cmdline 2>/dev/null)
  case "$cmd" in *"$TAG"*) ;; *) continue;; esac
  ppid=$(awk '/^PPid:/{print $2}' $d/status 2>/dev/null)
  rss=$(awk '/^VmRSS:/{print $2" "$3}' $d/status 2>/dev/null)
  thr=$(awk '/^Threads:/{print $2}' $d/status 2>/dev/null)
  echo "PID $pid PPID $ppid RSS $rss THREADS $thr CMD $cmd"
  PIDS="$PIDS $pid"
  if [ -z "$MAINPID" ]; then MAINPID=$pid; fi
done
echo "matched pids:$PIDS"
echo "--- OTHER BIGEVAL PROCS (context, DO NOT TOUCH) ---"
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  comm=$(cat $d/comm 2>/dev/null) || continue
  case "$comm" in *python*) ;; *) continue;; esac
  cmd=$(tr '\0' ' ' < $d/cmdline 2>/dev/null)
  case "$cmd" in *bigeval*) ;; *) continue;; esac
  case "$cmd" in *"$TAG"*) continue;; esac
  echo "OTHER PID $pid CMD $cmd"
done
echo "--- THREAD ENV (one exam worker) ---"
if [ -n "$PIDS" ]; then
  WPID=$(echo $PIDS | awk '{print $NF}')
  echo "env from pid: $WPID"
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|VECLIB_MAXIMUM_THREADS|BLIS_NUM_THREADS' || echo "NO_NUMPY_THREAD_ENV_SET"
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E '^EVO_WORKERS=' || echo "EVO_WORKERS_NOT_IN_ENVIRON"
fi
CWD=$(readlink /proc/$MAINPID/cwd 2>/dev/null)
echo "main cwd: $CWD"
F=/root/carbon/data/bigeval_live/$TAG.jsonl
echo "--- LINES BEFORE STOP ---"
if [ -f "$F" ]; then wc -l < "$F"; else echo "JSONL MISSING: $F"; fi
echo "--- STOPPING (SIGTERM) ---"
for pid in $PIDS; do
  if [ -d /proc/$pid ]; then kill -TERM $pid 2>/dev/null && echo "TERM sent to $pid"; fi
done
sleep 6
SURV=""
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  comm=$(cat $d/comm 2>/dev/null) || continue
  case "$comm" in *python*) ;; *) continue;; esac
  cmd=$(tr '\0' ' ' < $d/cmdline 2>/dev/null)
  case "$cmd" in *"$TAG"*) SURV="$SURV $pid";; *) continue;; esac
done
echo "survivors after TERM:$SURV"
if [ -n "$SURV" ]; then
  for pid in $SURV; do kill -KILL $pid 2>/dev/null && echo "KILL sent to $pid"; done
  sleep 3
fi
echo "--- POST-STOP VERIFY ---"
LEFT=""
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  comm=$(cat $d/comm 2>/dev/null) || continue
  case "$comm" in *python*) ;; *) continue;; esac
  cmd=$(tr '\0' ' ' < $d/cmdline 2>/dev/null)
  case "$cmd" in *"$TAG"*) LEFT="$LEFT $pid";; *) continue;; esac
done
if [ -z "$LEFT" ]; then echo "ZERO_RESIDUE for tag $TAG"; else echo "STILL RUNNING:$LEFT"; fi
ps -eo pid,ppid,etime,args | grep 'bigeval' | grep -v grep || echo "NO BIGEVAL PROCS AT ALL"
echo "--- LINES AFTER STOP ---"
if [ -f "$F" ]; then wc -l < "$F"; tail -1 "$F" | head -c 300; echo; fi
tail -3 /root/carbon/data/$TAG.out 2>/dev/null || true
tail -3 /root/carbon/data/m3s2g1_cal.launcher.log 2>/dev/null || true
echo "--- BIGEVAL SEED LOGIC ---"
BEV="$CWD/bigeval.py"
[ -f "$BEV" ] || BEV=/root/carbon/code/bigeval.py
[ -f "$BEV" ] || BEV=$(find /root/carbon -maxdepth 3 -name 'bigeval.py' 2>/dev/null | head -1)
echo "bigeval path: $BEV"
md5sum "$BEV" 2>/dev/null
grep -n -i 'seed' "$BEV" 2>/dev/null | head -50
echo "--- seed context lines ---"
grep -n -i -B2 -A2 'seed' "$BEV" 2>/dev/null | head -100
echo "${MARK}_END"
