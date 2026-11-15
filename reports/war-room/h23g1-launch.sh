#!/bin/bash
# h23: launch g1 calibration as a chained pair (field1 resume 89 + field2 200),
# full-load recipe: EVO_WORKERS=62, all numpy thread vars pinned to 1.
MARK="H23G1LAUNCH1954B"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- PRE-FLIGHT ---"
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running"
  ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
  echo "${MARK}_END"; exit 0
fi
echo "no bigeval running"
EV=$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)
BV=$(md5sum /root/carbon/code/bigeval.py | cut -d' ' -f1)
G1=$(md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz | cut -d' ' -f1)
E0W=$(md5sum /root/carbon/data/weights/cnet2_m3w3L.npz | cut -d' ' -f1)
E0P=$(md5sum /root/carbon/code/exp/echo_e0.py | cut -d' ' -f1)
echo "eval_v53=$EV bigeval=$BV g1=$G1 m3w3=$E0W echo_e0=$E0P"
[ "$EV" = "6d24da0eeeb28bad350d8b38f08ec3fc" ] || { echo "ABORT eval_v53 md5"; echo "${MARK}_END"; exit 1; }
[ "$BV" = "a5822338d979abea807fbd2ab4f7dace" ] || { echo "ABORT bigeval md5"; echo "${MARK}_END"; exit 1; }
[ "$G1" = "975c518c10198723fec7bfa57744d9c3" ] || { echo "ABORT g1 weights md5"; echo "${MARK}_END"; exit 1; }
[ "$E0W" = "e68d2660bab36bbee1ccd1a693cd67f0" ] || { echo "ABORT m3w3 weights md5"; echo "${MARK}_END"; exit 1; }
[ "$E0P" = "c969f7e7af94b3b8a2ea007044e80e00" ] || { echo "ABORT echo_e0 md5"; echo "${MARK}_END"; exit 1; }
echo "all md5 OK"
OLD=$(wc -l < /root/carbon/data/bigeval_live/m3s2g1_m2.jsonl)
echo "old field lines: $OLD (expect 111)"
[ "$OLD" = "111" ] || { echo "ABORT old field line count"; echo "${MARK}_END"; exit 1; }
ls /root/carbon/data/bigeval_live/m3s2g1_m2r.jsonl /root/carbon/data/bigeval_live/m3s2g1_e0.jsonl 2>&1 && { echo "ABORT new tag files already exist"; echo "${MARK}_END"; exit 1; }
echo "--- WRITE CHAIN RUNNER ---"
cat > /root/carbon/data/g1_chain_runner.sh <<'CHAIN'
#!/bin/bash
# chained: field1 resume (998111, 89) then field2 (998200, 200). Sequential, no idle gap.
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "CHAIN_START $(date '+%F %T %Z') field1" >> /root/carbon/data/g1_chain.status
../.venv/bin/python bigeval.py exp/m3s2g1.py exp/opp_master2r.py 998111 89 m3s2g1_m2r >> /root/carbon/data/m3s2g1_m2r.launcher.log 2>&1
echo "CHAIN_FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/g1_chain.status
echo "CHAIN_START $(date '+%F %T %Z') field2" >> /root/carbon/data/g1_chain.status
../.venv/bin/python bigeval.py exp/m3s2g1.py exp/echo_e0.py 998200 200 m3s2g1_e0 >> /root/carbon/data/m3s2g1_e0.launcher.log 2>&1
echo "CHAIN_FIELD2_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/g1_chain.status
CHAIN
chmod +x /root/carbon/data/g1_chain_runner.sh
echo "--- LAUNCH CHAIN (detached) ---"
setsid nohup /bin/bash /root/carbon/data/g1_chain_runner.sh >> /root/carbon/data/g1_chain.out 2>&1 < /dev/null &
echo "chain launcher pid: $!"
sleep 90
echo "--- VERIFY FIELD1 RUNNING ---"
ps -eo pid,ppid,etime,args | grep 'm3s2g1_m2r' | grep -v grep || echo "NO FIELD1 PROC - LAUNCH FAILED"
echo "bigeval proc count: $(ps -eo args | grep -c '[b]igeval.py exp/m3s2g1.py')"
WPID=$(pgrep -f 'bigeval.py exp/m3s2g1.py' | tail -1)
echo "sample worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'EVO_WORKERS|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS' || echo "THREAD ENV NOT VISIBLE"
  awk '/^Threads:/{print "worker threads:", $2}' /proc/$WPID/status 2>/dev/null
fi
echo "--- LIVE FILE ---"
ls -la /root/carbon/data/bigeval_live/m3s2g1_m2r.jsonl 2>&1
wc -l /root/carbon/data/bigeval_live/m3s2g1_m2r.jsonl 2>&1
cat /proc/loadavg
echo "${MARK}_END"
