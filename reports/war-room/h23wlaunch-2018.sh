#!/bin/bash
# h23: launch Line A paired calibration chain (4 fields x 200 games, full-load recipe).
# wr1 = weighted attention replica of Master3s2; wg1 = weighted control replica.
# Fields: wr1 vs Master2r (996000), wr1 vs Echo2 (999000), wg1 vs Master2r (996000), wg1 vs Echo2 (999000).
MARK="H23WLAUNCH2018D"
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
WR=$(md5sum /root/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz | cut -d' ' -f1)
WG=$(md5sum /root/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz | cut -d' ' -f1)
W5=$(md5sum /root/carbon/data/weights/cnet2_m3w5L.npz | cut -d' ' -f1)
E2P=$(md5sum /root/carbon/code/exp/echo2.py | cut -d' ' -f1)
echo "eval_v53=$EV bigeval=$BV wr1=$WR wg1=$WG m3w5=$W5 echo2py=$E2P"
[ "$EV" = "6d24da0eeeb28bad350d8b38f08ec3fc" ] || { echo "ABORT eval_v53 md5"; echo "${MARK}_END"; exit 1; }
[ "$BV" = "a5822338d979abea807fbd2ab4f7dace" ] || { echo "ABORT bigeval md5"; echo "${MARK}_END"; exit 1; }
[ "$WR" = "5db492c2d2a6412c40a14fe1d710eec2" ] || { echo "ABORT wr1 md5"; echo "${MARK}_END"; exit 1; }
[ "$WG" = "a0b2171c7575c81e4befca76a3739d36" ] || { echo "ABORT wg1 md5"; echo "${MARK}_END"; exit 1; }
[ "$W5" = "82fde46b2633950d3cbb29e8cacbfdc2" ] || { echo "ABORT m3w5 md5"; echo "${MARK}_END"; exit 1; }
[ "$E2P" = "15b6d754e6e21de33870074d3983a16f" ] || { echo "ABORT echo2.py md5"; echo "${MARK}_END"; exit 1; }
echo "all md5 OK"
for f in m3s2wr1_m2 m3s2wr1_echo2 m3s2wg1_m2 m3s2wg1_echo2; do
  if [ -e /root/carbon/data/bigeval_live/$f.jsonl ]; then
    echo "ABORT tag file already exists: $f.jsonl"; echo "${MARK}_END"; exit 1
  fi
done
echo "new tag files absent"
cd /root/carbon/code
echo "--- WEIGHT KEY INVENTORY ---"
../.venv/bin/python - <<'PYEOF'
import numpy as np
for name in ("cnet2_m3s2wr1_fromstudio.npz", "cnet2_m3s2wg1_fromstudio.npz"):
    d = np.load("/root/carbon/data/weights/" + name)
    keys = list(d.keys())
    n = sum(int(d[k].size) for k in keys)
    attn = [k for k in keys if "att" in k.lower() or "ta_" in k.lower() or "_ta" in k.lower()]
    print(name, "keys", len(keys), "params", n, "attn_like_keys", len(attn))
PYEOF
echo "--- WRITE WRAPPERS (adapt g1/r1 by weights swap only) ---"
sed 's/cnet2_m3s2g1L\.npz/cnet2_m3s2wg1_fromstudio.npz/' /root/carbon/code/exp/m3s2g1.py > /root/carbon/code/exp/m3s2wg1.py
sed 's/cnet2_m3s2r1L\.npz/cnet2_m3s2wr1_fromstudio.npz/' /root/carbon/code/exp/m3s2r1.py > /root/carbon/code/exp/m3s2wr1.py
grep -H '_W = ' /root/carbon/code/exp/m3s2wr1.py /root/carbon/code/exp/m3s2wg1.py
diff /root/carbon/code/exp/m3s2g1.py /root/carbon/code/exp/m3s2wg1.py
diff /root/carbon/code/exp/m3s2r1.py /root/carbon/code/exp/m3s2wr1.py
../.venv/bin/python -c "import ast; ast.parse(open('/root/carbon/code/exp/m3s2wr1.py').read()); ast.parse(open('/root/carbon/code/exp/m3s2wg1.py').read()); print('wrapper syntax OK')"
echo "--- WRITE CHAIN RUNNER ---"
cat > /root/carbon/data/lineA_cal_runner.sh <<'CHAIN'
#!/bin/bash
# chained Line A paired calibration: 4 fields x 200, no idle gap. Full-load recipe.
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "CHAIN_START $(date '+%F %T %Z') field1 m3s2wr1_m2" >> /root/carbon/data/lineA_cal.status
../.venv/bin/python bigeval.py exp/m3s2wr1.py exp/opp_master2r.py 996000 200 m3s2wr1_m2 >> /root/carbon/data/m3s2wr1_m2.launcher.log 2>&1
echo "CHAIN_FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/lineA_cal.status
echo "CHAIN_START $(date '+%F %T %Z') field2 m3s2wr1_echo2" >> /root/carbon/data/lineA_cal.status
../.venv/bin/python bigeval.py exp/m3s2wr1.py exp/echo2.py 999000 200 m3s2wr1_echo2 >> /root/carbon/data/m3s2wr1_echo2.launcher.log 2>&1
echo "CHAIN_FIELD2_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/lineA_cal.status
echo "CHAIN_START $(date '+%F %T %Z') field3 m3s2wg1_m2" >> /root/carbon/data/lineA_cal.status
../.venv/bin/python bigeval.py exp/m3s2wg1.py exp/opp_master2r.py 996000 200 m3s2wg1_m2 >> /root/carbon/data/m3s2wg1_m2.launcher.log 2>&1
echo "CHAIN_FIELD3_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/lineA_cal.status
echo "CHAIN_START $(date '+%F %T %Z') field4 m3s2wg1_echo2" >> /root/carbon/data/lineA_cal.status
../.venv/bin/python bigeval.py exp/m3s2wg1.py exp/echo2.py 999000 200 m3s2wg1_echo2 >> /root/carbon/data/m3s2wg1_echo2.launcher.log 2>&1
echo "CHAIN_ALL_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/lineA_cal.status
CHAIN
chmod +x /root/carbon/data/lineA_cal_runner.sh
echo "--- LAUNCH CHAIN (detached) ---"
setsid nohup /bin/bash /root/carbon/data/lineA_cal_runner.sh >> /root/carbon/data/lineA_cal.out 2>&1 < /dev/null &
echo "chain launcher pid: $!"
sleep 90
echo "--- VERIFY ---"
cat /root/carbon/data/lineA_cal.status 2>&1
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep || echo "NO_BIGEVAL_PROC_AT_VERIFY"
echo "bigeval proc count: $(ps -eo args | grep -c '[b]igeval.py exp/')"
WPID=$(pgrep -f 'bigeval.py exp/' | tail -1)
echo "sample worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'EVO_WORKERS|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS' || echo "THREAD ENV NOT VISIBLE"
  awk '/^Threads:/{print "worker threads:", $2}' /proc/$WPID/status 2>/dev/null
fi
echo "--- LIVE FILES ---"
wc -l /root/carbon/data/bigeval_live/m3s2wr1_m2.jsonl /root/carbon/data/bigeval_live/m3s2wr1_echo2.jsonl /root/carbon/data/bigeval_live/m3s2wg1_m2.jsonl /root/carbon/data/bigeval_live/m3s2wg1_echo2.jsonl 2>&1
cat /proc/loadavg
echo "${MARK}_END"
