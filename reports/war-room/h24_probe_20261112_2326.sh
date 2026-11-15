#!/bin/bash
# H24 read-only probe 2026-11-12 23:15: spot-28 K4 probe stuck check.
# Marker: H24PROBE2326. READ-ONLY: no kill, no start, no file change.
echo "H24PROBE2326_BEGIN"
date -u "+UTC %F %T"; date "+CST %F %T %Z"
echo "--- load ---"
timeout 10 uptime
echo "--- bigeval/eval processes ---"
timeout 10 ps -eo pid,etime,pcpu,args | grep -E "bigeval|eval_v53|s28k4|k4v2" | grep -v grep || echo "NO_MATCH"
echo "--- all python procs ---"
timeout 10 ps -eo pid,etime,args | grep python3 | grep -v grep | head -20 || echo "NO_PYTHON"
echo "--- live file on machine ---"
timeout 20 find /root/carbon -name "k4v2r1_e0.jsonl" 2>/dev/null | while read f; do echo "FILE $f"; timeout 10 ls -l --time-style=full-iso "$f"; timeout 10 wc -l "$f"; done
echo "--- snapshot/sync loop procs ---"
timeout 10 ps -eo pid,etime,args | grep -E "sync|snapshot|git push" | grep -v grep | head -10 || echo "NO_SYNC_PROC"
echo "--- disk ---"
timeout 10 df -h /root/carbon | tail -1
echo "H24PROBE2326_END"
