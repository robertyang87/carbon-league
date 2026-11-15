#!/bin/bash
# h23 read-only parallelism probe: cores, mem, worker RSS/threads, numpy thread env, bigeval seeding.
MARK="H23PAR1939K7"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
hostname
echo "--- CPU ---"
echo "nproc: $(nproc)"
grep -m1 'model name' /proc/cpuinfo
echo "processor entries: $(grep -c '^processor' /proc/cpuinfo)"
cat /proc/loadavg
echo "--- MEM ---"
free -m
echo "--- BIGEVAL PROCS (ps) ---"
ps -eo pid,ppid,etime,args | grep 'bigeval' | grep -v grep || echo "NO BIGEVAL PROC"
echo "--- PER-PROC RSS/THREADS (comm=python only) ---"
MAINPID=""
for pid in $(pgrep -f bigeval); do
  comm=$(cat /proc/$pid/comm 2>/dev/null)
  case "$comm" in *python*) ;; *) continue;; esac
  ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status 2>/dev/null)
  rss=$(awk '/^VmRSS:/{print $2" "$3}' /proc/$pid/status 2>/dev/null)
  thr=$(awk '/^Threads:/{print $2}' /proc/$pid/status 2>/dev/null)
  echo "PID $pid PPID $ppid RSS $rss THREADS $thr"
  if [ -z "$MAINPID" ]; then MAINPID=$pid; fi
done
echo "--- THREAD ENV (of last python bigeval pid) ---"
WPID=""
for pid in $(pgrep -f bigeval); do
  comm=$(cat /proc/$pid/comm 2>/dev/null)
  case "$comm" in *python*) WPID=$pid;; *) continue;; esac
done
echo "worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|VECLIB_MAXIMUM_THREADS|BLIS_NUM_THREADS' || echo "NO_NUMPY_THREAD_ENV_SET"
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E '^EVO_WORKERS=' || echo "EVO_WORKERS not in environ (likely CLI/default)"
fi
echo "--- BIGEVAL SEED LOGIC ---"
CWD=$(readlink /proc/$MAINPID/cwd 2>/dev/null)
echo "main cwd: $CWD"
BEV="$CWD/bigeval.py"
[ -f "$BEV" ] || BEV=/root/carbon/code/bigeval.py
[ -f "$BEV" ] || BEV=$(find /root/carbon -maxdepth 3 -name 'bigeval.py' 2>/dev/null | head -1)
echo "bigeval path: $BEV"
md5sum "$BEV" 2>/dev/null
grep -n -i 'seed' "$BEV" 2>/dev/null | head -50
echo "--- seed context lines ---"
grep -n -i -B2 -A2 'seed' "$BEV" 2>/dev/null | head -100
echo "${MARK}_END"
