#!/bin/bash
# H28: K4 probe on spot-28. Recipe follows the proven h23 20:42 launch
# (pre-flight md5s, byte-identical wrapper recreation, CONF read-back,
# serial chain r1 -> g1, full-load env), retargeted to spot-28.
# Marker: S28K4LAUNCH2252
MARK="S28K4LAUNCH2252"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
echo "--- RESIDUAL SCAN ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval|snaploop|sync' | grep -v grep || echo NO-RESIDUAL-PROC
if pgrep -f 'bigeval.py exp/' > /dev/null; then
  echo "ABORT: a bigeval field is already running"
  ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
  echo "${MARK}_END"; exit 0
fi
echo "no bigeval running"
echo "--- ONBOX SYNC CHECK ---"
ls /root/carbon/*sync* /root/carbon/data/*sync* 2>/dev/null || echo NO-SYNC-FILES
crontab -l 2>/dev/null | head -5 || echo NO-CRONTAB
echo "--- PRE-FLIGHT MD5 ---"
EV=$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)
BV=$(md5sum /root/carbon/code/bigeval.py | cut -d' ' -f1)
R1=$(md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz | cut -d' ' -f1)
G1=$(md5sum /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz | cut -d' ' -f1)
E0=$(md5sum /root/carbon/data/weights/cnet2_m3w3L.npz | cut -d' ' -f1)
echo "eval_v53=$EV bigeval=$BV k4v2r1=$R1 k4v2g1=$G1 e0w=$E0"
[ "$EV" = "6d24da0eeeb28bad350d8b38f08ec3fc" ] || { echo "ABORT eval_v53 md5"; echo "${MARK}_END"; exit 1; }
[ "$BV" = "a5822338d979abea807fbd2ab4f7dace" ] || { echo "ABORT bigeval md5"; echo "${MARK}_END"; exit 1; }
[ "$R1" = "344cfde939debb0de7f8834af069ea16" ] || { echo "ABORT k4v2r1 weight md5"; echo "${MARK}_END"; exit 1; }
[ "$G1" = "a594efd6cdb865d978e511c66d299206" ] || { echo "ABORT k4v2g1 weight md5"; echo "${MARK}_END"; exit 1; }
[ "$E0" = "e68d2660bab36bbee1ccd1a693cd67f0" ] || { echo "ABORT E0 weight md5"; echo "${MARK}_END"; exit 1; }
echo "all md5 OK"
echo "--- TAG FILES ---"
for f in k4v2r1_e0 k4v2g1_e0; do
  P=/root/carbon/data/bigeval_live/$f.jsonl
  if [ -e "$P" ]; then
    SZ=$(stat -c%s "$P")
    echo "exists $f.jsonl size=$SZ"
    if [ "$SZ" != "0" ]; then echo "ABORT tag file non-empty: $f.jsonl"; echo "${MARK}_END"; exit 1; fi
    rm -f "$P"; echo "removed empty stale $f.jsonl"
  else
    echo "absent $f.jsonl (expected)"
  fi
done
echo "--- RECREATE WRAPPERS (byte-identical heredocs) ---"
cat > /root/carbon/code/exp/k4v2r1.py <<'PYEOF'
"""k4v2r1: K4TacSearch-v2 attention replica (cnet2_k4v2r1_fromstudio.npz), SEARCH ON.
K4 probe arm: identical SEARCH config to exp/k4r1_on.py (pack-frozen), weights swapped to the v2 attention replica."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_k4v2r1_fromstudio.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
PYEOF
cat > /root/carbon/code/exp/k4v2g1.py <<'PYEOF'
"""k4v2g1: K4TacSearch-v2 control replica (cnet2_k4v2g1_fromstudio.npz), SEARCH ON.
K4 probe arm: identical SEARCH config to exp/k4r1_on.py (pack-frozen), weights swapped to the v2 control replica."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_k4v2g1_fromstudio.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
PYEOF
WR1=$(md5sum /root/carbon/code/exp/k4v2r1.py | cut -d' ' -f1)
WG1=$(md5sum /root/carbon/code/exp/k4v2g1.py | cut -d' ' -f1)
echo "wrapper md5s: k4v2r1.py=$WR1 k4v2g1.py=$WG1"
[ "$WR1" = "c395faa8f3d783c509cb2f9fc82db676" ] || { echo "ABORT k4v2r1.py md5 mismatch"; echo "${MARK}_END"; exit 1; }
[ "$WG1" = "d384dfaa35adfe3fd12ebe291e275108" ] || { echo "ABORT k4v2g1.py md5 mismatch"; echo "${MARK}_END"; exit 1; }
echo "wrapper md5s OK"
/root/carbon/.venv/bin/python -m py_compile /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py && echo "py_compile OK"
echo "--- CONF READ-BACK ---"
cd /root/carbon/code
../.venv/bin/python - <<'PYEOF'
import importlib.util
FROZEN = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}
for name in ("k4v2r1", "k4v2g1"):
    spec = importlib.util.spec_from_file_location(name, f"/root/carbon/code/exp/{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    ok = (m.CONF == FROZEN)
    print(name, "CONF ==", m.CONF)
    print(name, "SEARCH =", m.CONF.get('SEARCH'), "MATCHES_FROZEN =", ok)
    assert ok, f"{name} CONF mismatch"
print("CONF_READBACK_ALL_OK")
PYEOF
[ $? -eq 0 ] || { echo "ABORT CONF read-back"; echo "${MARK}_END"; exit 1; }
echo "--- WRITE CHAIN RUNNER ---"
cat > /root/carbon/data/k4probe_runner.sh <<'CHAIN'
#!/bin/bash
cd /root/carbon/code
export EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "CHAIN_START $(date '+%F %T %Z') field1 k4v2r1_e0" >> /root/carbon/data/k4probe.status
../.venv/bin/python bigeval.py exp/k4v2r1.py exp/echo_e0.py 997000 200 k4v2r1_e0 >> /root/carbon/data/k4v2r1_e0.launcher.log 2>&1
echo "CHAIN_FIELD1_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/k4probe.status
echo "CHAIN_START $(date '+%F %T %Z') field2 k4v2g1_e0" >> /root/carbon/data/k4probe.status
../.venv/bin/python bigeval.py exp/k4v2g1.py exp/echo_e0.py 997000 200 k4v2g1_e0 >> /root/carbon/data/k4v2g1_e0.launcher.log 2>&1
echo "CHAIN_ALL_DONE rc=$? $(date '+%F %T %Z')" >> /root/carbon/data/k4probe.status
CHAIN
chmod +x /root/carbon/data/k4probe_runner.sh
rm -f /root/carbon/data/k4probe.status
echo "--- LAUNCH CHAIN (detached) ---"
setsid nohup /bin/bash /root/carbon/data/k4probe_runner.sh >> /root/carbon/data/k4probe.out 2>&1 < /dev/null &
echo "chain launcher pid: $!"
sleep 90
echo "--- VERIFY ---"
cat /root/carbon/data/k4probe.status 2>&1
echo "bigeval proc count: $(ps -eo args | grep -c '[b]igeval.py exp/')"
WPID=$(pgrep -f 'bigeval.py exp/' | tail -1)
echo "sample worker pid: $WPID"
if [ -n "$WPID" ]; then
  tr '\0' '\n' < /proc/$WPID/environ 2>/dev/null | grep -E 'EVO_WORKERS|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS' || echo "THREAD ENV NOT VISIBLE"
  awk '/^Threads:/{print "worker threads:", $2}' /proc/$WPID/status 2>/dev/null
fi
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>&1
head -1 /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>/dev/null | cut -c1-200
tail -1 /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl 2>/dev/null | cut -c1-200
cat /proc/loadavg
free -m | head -2
echo "${MARK}_END"
