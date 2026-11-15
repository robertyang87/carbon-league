echo K4LAUNCH2006_BEGIN
date "+%F %T %Z"
echo "--- PRE-FLIGHT ---"
md5sum /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py /root/carbon/code/exp/echo_e0.py
ps -eo pid,args | grep bigeval | grep -v grep || echo NO-BIGEVAL-PROC
cat > /root/carbon/data/k4v2_chain_runner.sh <<'CHAINEOF'
#!/bin/bash
cd /root/carbon
echo "CHAIN_START $(date '+%F %T %Z') field1 k4v2r1_e0" > /root/carbon/data/k4v2_chain.status
EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /root/carbon/.venv/bin/python code/bigeval.py exp/k4v2r1.py exp/echo_e0.py 997000 200 k4v2r1_e0 > /root/carbon/data/k4v2r1_e0.launcher.log 2>&1
echo "FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/k4v2_chain.status
EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /root/carbon/.venv/bin/python code/bigeval.py exp/k4v2g1.py exp/echo_e0.py 997000 200 k4v2g1_e0 > /root/carbon/data/k4v2g1_e0.launcher.log 2>&1
echo "FIELD2_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/k4v2_chain.status
echo "CHAIN_DONE $(date '+%F %T %Z')" >> /root/carbon/data/k4v2_chain.status
CHAINEOF
chmod +x /root/carbon/data/k4v2_chain_runner.sh
rm -f /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl
nohup /root/carbon/data/k4v2_chain_runner.sh > /root/carbon/data/k4v2_chain.log 2>&1 &
echo "CHAIN_PID $!"
sleep 30
echo "--- PROCS ---"
pgrep -fc "bigeval.py exp/k4v2r1.py exp/echo_e0.py" || true
ps -eo pid,ppid,etime,args | grep -E "bigeval.py exp/k4v2r1.py" | grep -v grep | head -3
echo "--- JSONL ---"
ls -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>/dev/null || echo NO-JSONL-YET
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>/dev/null || true
echo "--- STATUS ---"
cat /root/carbon/data/k4v2_chain.status
cat /proc/loadavg
echo K4LAUNCH2006_END
