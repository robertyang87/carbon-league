#!/bin/bash
# C1: launch Fable r1 calibration pair on spot-29 (vs Master2r 996000,
# then vs Echo2 999000), full-load recipe. Adapted from h23 m7bsr1 launch.
MARK="S29FLAUNCH2259"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- RESIDUAL SCAN ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval|snaploop' | grep -v grep || echo NO-RESIDUAL-PROC
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running"
  ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
  echo "${MARK}_END"; exit 0
fi
echo "no bigeval running"
echo "--- PRE-FLIGHT MD5 ---"
EV=$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)
BV=$(md5sum /root/carbon/code/bigeval.py | cut -d' ' -f1)
FB=$(md5sum /root/carbon/data/weights/cnet2_fable_r1_fromstudio.npz | cut -d' ' -f1)
W5=$(md5sum /root/carbon/data/weights/cnet2_m3w5L.npz | cut -d' ' -f1)
E2P=$(md5sum /root/carbon/code/exp/echo2.py | cut -d' ' -f1)
TPL=$(md5sum /root/carbon/code/exp/m3s2r1.py | cut -d' ' -f1)
echo "eval_v53=$EV bigeval=$BV fable_r1=$FB m3w5=$W5 echo2py=$E2P template=$TPL"
[ "$EV" = "6d24da0eeeb28bad350d8b38f08ec3fc" ] || { echo "ABORT eval_v53 md5"; echo "${MARK}_END"; exit 1; }
[ "$BV" = "a5822338d979abea807fbd2ab4f7dace" ] || { echo "ABORT bigeval md5"; echo "${MARK}_END"; exit 1; }
[ "$FB" = "8bec64d7618686f52f73d592e7099940" ] || { echo "ABORT fable_r1 md5"; echo "${MARK}_END"; exit 1; }
[ "$W5" = "82fde46b2633950d3cbb29e8cacbfdc2" ] || { echo "ABORT m3w5 md5"; echo "${MARK}_END"; exit 1; }
[ "$E2P" = "15b6d754e6e21de33870074d3983a16f" ] || { echo "ABORT echo2.py md5"; echo "${MARK}_END"; exit 1; }
[ "$TPL" = "73542cdd0ea50dce87e3638fd4c9bf5a" ] || { echo "ABORT template md5"; echo "${MARK}_END"; exit 1; }
echo "all md5 OK"
for f in fabler1_m2 fabler1_echo2; do
  if [ -e /root/carbon/data/bigeval_live/$f.jsonl ]; then
    echo "ABORT tag file already exists: $f.jsonl"; echo "${MARK}_END"; exit 1
  fi
done
echo "new tag files absent"
echo "--- WEIGHT KEY INVENTORY ---"
cd /root/carbon/code
../.venv/bin/python - <<'PYEOF'
import numpy as np
d = np.load("/root/carbon/data/weights/cnet2_fable_r1_fromstudio.npz")
keys = list(d.keys())
print("fable_r1 keys", len(keys), "params", sum(int(d[k].size) for k in keys))
PYEOF
echo "--- WRITE WRAPPER (template m3s2r1.py, weights swap only) ---"
sed 's/cnet2_m3s2r1L\.npz/cnet2_fable_r1_fromstudio.npz/' /root/carbon/code/exp/m3s2r1.py > /root/carbon/code/exp/fabler1.py
grep -H '_W = ' /root/carbon/code/exp/fabler1.py
diff /root/carbon/code/exp/m3s2r1.py /root/carbon/code/exp/fabler1.py
grep -q 'cnet2_fable_r1_fromstudio.npz' /root/carbon/code/exp/fabler1.py || { echo "ABORT wrapper swap failed"; echo "${MARK}_END"; exit 1; }
../.venv/bin/python -m py_compile /root/carbon/code/exp/fabler1.py && echo "py_compile OK"
echo "--- WRITE CHAIN RUNNER ---"
cat > /root/carbon/data/fabler1_cal_runner.sh <<'CHAIN'
#!/bin/bash
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "CHAIN_START $(date '+%F %T %Z') field1 fabler1_m2" >> /root/carbon/data/fabler1_cal.status
../.venv/bin/python bigeval.py exp/fabler1.py exp/opp_master2r.py 996000 200 fabler1_m2 >> /root/carbon/data/fabler1_m2.launcher.log 2>&1
echo "CHAIN_FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/fabler1_cal.status
echo "CHAIN_START $(date '+%F %T %Z') field2 fabler1_echo2" >> /root/carbon/data/fabler1_cal.status
../.venv/bin/python bigeval.py exp/fabler1.py exp/echo2.py 999000 200 fabler1_echo2 >> /root/carbon/data/fabler1_echo2.launcher.log 2>&1
echo "CHAIN_ALL_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/fabler1_cal.status
CHAIN
chmod +x /root/carbon/data/fabler1_cal_runner.sh
rm -f /root/carbon/data/fabler1_cal.status
echo "--- LAUNCH CHAIN (detached) ---"
setsid nohup /bin/bash /root/carbon/data/fabler1_cal_runner.sh >> /root/carbon/data/fabler1_cal.out 2>&1 < /dev/null &
echo "chain launcher pid: $!"
sleep 90
echo "--- VERIFY ---"
cat /root/carbon/data/fabler1_cal.status 2>&1
echo "bigeval proc count: $(ps -eo args | grep -c '[b]igeval.py exp/')"
WPID=$(pgrep -f 'bigeval.py exp/' | tail -1)
echo "sample worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'EVO_WORKERS|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS' || echo "THREAD ENV NOT VISIBLE"
  awk '/^Threads:/{print "worker threads:", $2}' /proc/$WPID/status 2>/dev/null
fi
wc -l /root/carbon/data/bigeval_live/fabler1_m2.jsonl 2>&1
head -1 /root/carbon/data/bigeval_live/fabler1_m2.jsonl 2>/dev/null | cut -c1-200
cat /proc/loadavg
free -m | head -2
echo "${MARK}_END"
