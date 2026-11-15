"""Generate s30echo3-timing.sh embedding (a) the C03 timing tool fetched
from repo main and (b) a detailed step probe adapted from n1r_timing.py
(pack -> echo3_int8.py, opponent -> opp_master2r.py, marker E3TIME_JSON).
Asserts the tool md5 before embedding."""
import base64, hashlib

tool = open("/tmp/cnet2_search_timing.py", "rb").read()
md5 = hashlib.md5(tool).hexdigest()
assert md5 == "c60390a36cc6e895eb9a725f89f5973c", md5

probe_src = open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/n1r_timing.py").read()
probe_src = probe_src.replace("n1r_int8.py", "echo3_int8.py")
probe_src = probe_src.replace("echo_e0.py", "opp_master2r.py")
probe_src = probe_src.replace("N1RTIME_JSON", "E3TIME_JSON")
assert "echo3_int8.py" in probe_src and "opp_master2r.py" in probe_src and "E3TIME_JSON" in probe_src
probe = probe_src.encode()

script = """#!/bin/bash
MARK=S30E3TIME0315
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
echo -n "bigeval-procs: "; pgrep -cf "bigeval.py" || echo 0
cat /proc/loadavg
cd /root/carbon/code || { echo NO-CODE-DIR; echo "$MARK PAYLOAD-END"; exit 0; }
echo "__TOOLB64__" | base64 -d > /root/carbon/code/cnet2_search_timing.py
echo "__PROBEB64__" | base64 -d > /root/carbon/code/e3_timing_probe.py
echo "TOOL-MD5=$(md5sum /root/carbon/code/cnet2_search_timing.py | awk '{print $1}')"
/root/carbon/.venv/bin/python -m py_compile /root/carbon/code/cnet2_search_timing.py /root/carbon/code/e3_timing_probe.py && echo PYCOMPILE-OK
echo "--- CGROUP-SETUP ---"
CG=/sys/fs/cgroup/e3timing
mkdir -p "$CG" 2>/dev/null
echo "150000 100000" > "$CG/cpu.max" 2>/dev/null || echo CPU-MAX-WRITE-FAIL
echo 536870912 > "$CG/memory.max" 2>/dev/null || echo MEM-MAX-WRITE-FAIL
echo "cpu.max=$(cat $CG/cpu.max) memory.max=$(cat $CG/memory.max)"
echo "--- C03-TOOL-RUN (1 game, seed 700001, vs opp_master2r) ---"
T0=$(date +%s.%N)
bash -c 'echo $$ > /sys/fs/cgroup/e3timing/cgroup.procs; exec /root/carbon/.venv/bin/python /root/carbon/code/cnet2_search_timing.py ../data/packs/echo3_int8.py exp/opp_master2r.py 700001 1' > /tmp/e3tool.out 2> /tmp/e3tool.err
RC=$?
T1=$(date +%s.%N)
echo "TOOL-RC=$RC TOOL-WALL-S=$(echo "$T1 - $T0" | bc)"
cat /tmp/e3tool.out
echo "--- TOOL-STDERR ---"
cat /tmp/e3tool.err
echo "memory.peak(tool)=$(cat $CG/memory.peak 2>/dev/null)"
grep -E "nr_throttled|throttled_usec" "$CG/cpu.stat" 2>/dev/null
echo "--- DETAILED-PROBE-RUN (fresh cgroup) ---"
CG2=/sys/fs/cgroup/e3timing2
mkdir -p "$CG2" 2>/dev/null
echo "150000 100000" > "$CG2/cpu.max" 2>/dev/null
echo 536870912 > "$CG2/memory.max" 2>/dev/null
echo "cpu.max=$(cat $CG2/cpu.max) memory.max=$(cat $CG2/memory.max)"
bash -c 'echo $$ > /sys/fs/cgroup/e3timing2/cgroup.procs; exec /root/carbon/.venv/bin/python /root/carbon/code/e3_timing_probe.py' > /tmp/e3probe.out 2> /tmp/e3probe.err
echo "PROBE-RC=$?"
cat /tmp/e3probe.out
tail -3 /tmp/e3probe.err
echo "memory.peak(probe)=$(cat $CG2/memory.peak 2>/dev/null)"
rmdir "$CG" "$CG2" 2>/dev/null
echo "$MARK PAYLOAD-END"
"""
script = script.replace("__TOOLB64__", base64.b64encode(tool).decode())
script = script.replace("__PROBEB64__", base64.b64encode(probe).decode())
out = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/s30echo3-timing.sh"
open(out, "w").write(script)
print("wrote", out, len(script), "bytes; tool md5", md5)
