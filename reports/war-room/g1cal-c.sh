#!/bin/bash
# g1cal step C: verify new eval_v53.py landed (via DGX relay) + launch m3s2g1 vs Master2.
# Small script by design: cloud-assistant command content has a size limit
# (a 41KB script with an embedded file failed with CmdContent.ExceedLimit).
MARK="G1CALC5P9"
echo "MARKER $MARK START"
date '+%F %T %Z'
echo "--- PRE-LAUNCH BIGEVAL CHECK ---"
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running:"
  ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
  echo "MARKER $MARK END"
  exit 0
fi
echo "no bigeval running"
echo "--- VERIFY eval_v53.py (expect 6d24da0e...) ---"
md5sum /root/carbon/code/eval_v53.py
if [ "$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)" != "6d24da0eeeb28bad350d8b38f08ec3fc" ]; then
  echo "ABORT: eval_v53 md5 MISMATCH, not launching"
  echo "MARKER $MARK END"
  exit 1
fi
echo "eval_v53 md5 OK"
echo "--- VERIFY bigeval.py + wrapper + weights ---"
md5sum /root/carbon/code/bigeval.py /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/opp_master2r.py /root/carbon/data/weights/cnet2_m3s2g1L.npz
if [ "$(md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz | cut -d' ' -f1)" != "975c518c10198723fec7bfa57744d9c3" ]; then
  echo "ABORT: g1 weights md5 MISMATCH, not launching"
  echo "MARKER $MARK END"
  exit 1
fi
echo "weights md5 OK (= DGX ep19/last)"
echo "--- LAUNCH ---"
cd /root/carbon/code || { echo "CD FAIL"; echo "MARKER $MARK END"; exit 1; }
EVO_WORKERS=7 setsid nohup ../.venv/bin/python bigeval.py exp/m3s2g1.py exp/opp_master2r.py 998000 200 m3s2g1_m2 >> /root/carbon/data/m3s2g1_m2.out 2>&1 &
sleep 60
echo "--- NEW PROCS (m3s2g1_m2) ---"
ps -eo pid,ppid,etime,args | grep 'm3s2g1_m2' | grep -v grep || echo "NO NEW PROC - LAUNCH FAILED"
echo "--- WORKER COUNT ---"
ps -eo args | grep -c '[b]igeval.py exp/m3s2g1.py' || true
echo "--- LIVE FILE ---"
ls -la /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1
wc -l /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1
head -1 /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl 2>&1 | cut -c1-600
echo "--- OUT FILE HEAD ---"
head -5 /root/carbon/data/m3s2g1_m2.out 2>&1
echo "MARKER $MARK END"
