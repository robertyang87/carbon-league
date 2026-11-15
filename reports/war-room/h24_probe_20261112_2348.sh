#!/bin/bash
# H24 read-only probe, run 2026-11-12 ~23:48 CST. Marker: H24PROBE2348.
# Target: spot-28 (i-uf65egu9fey7xf7tu8ft), cn-shanghai. READ-ONLY.
echo "H24PROBE2348_BEGIN"
date -u "+UTC %F %T"; date "+CST %F %T %Z"
echo "--- load ---"
timeout 10 uptime
echo "--- bigeval/eval processes ---"
timeout 10 ps -eo pid,etime,pcpu,args | grep -E "bigeval|eval_v53|k4v2" | grep -v grep || echo "NO_MATCH"
echo "--- live files on machine ---"
timeout 20 find /root/carbon -name "k4v2*_e0.jsonl" 2>/dev/null | while read f; do echo "FILE $f"; timeout 10 ls -l --time-style=full-iso "$f"; timeout 10 wc -l "$f"; done
echo "--- g1 summary/cal files ---"
timeout 20 find /root/carbon -name "*k4v2g1*" -newer /root/carbon/data/bigeval.jsonl 2>/dev/null | head; timeout 10 ls -l --time-style=full-iso /root/carbon/data/ 2>/dev/null | tail -15
echo "--- snapshot loop procs (by script name) ---"
timeout 10 ps -eo pid,etime,args | grep -E "snaploop|s28k4" | grep -v grep || echo "NO_SNAPLOOP"
echo "--- disk ---"
timeout 10 df -h /root/carbon | tail -1
echo "H24PROBE2348_END"
