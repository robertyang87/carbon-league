#!/bin/bash
# H24 read-only probe, run 2026-11-12 ~23:59 CST. Marker: H24PROBE2359.
# Target: spot-28 (i-uf65egu9fey7xf7tu8ft), cn-shanghai. READ-ONLY.
echo "H24PROBE2359_BEGIN"
date -u "+UTC %F %T"; date "+CST %F %T %Z"
echo "--- machine bigeval summary (both arms) ---"
timeout 20 cat /root/carbon/data/bigeval.jsonl
echo "--- k4probe.status ---"
timeout 10 cat /root/carbon/data/k4probe.status
echo "--- g1 launcher log ---"
timeout 10 cat /root/carbon/data/k4v2g1_e0.launcher.log
echo "--- live file sizes ---"
timeout 10 ls -l --time-style=full-iso /root/carbon/data/bigeval_live/
echo "--- any python/snap processes ---"
timeout 10 ps -eo pid,etime,args | grep -E "python3|snap|bigeval" | grep -v grep || echo "NO_PY_SNAP"
echo "H24PROBE2359_END"
