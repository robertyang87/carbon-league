#!/bin/bash
# DGX read-only check for Echo3 _OPTS (war-room order, 01:56): search s26stage +
# data/packs for *n1r* pack files and copy back _OPTS + make_agent if found.
# Also closes smoke step-0 (repo-path md5s) and re-reads the two summary JSONs.
# Marker: DGXS26OPTS
MARK=DGXS26OPTS
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXS26OPTS DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo "--- s26stage top ---"
ls -la /home/arm/junyang/s26stage/ 2>&1 | head -40 || echo "S26STAGE-MISSING"
echo "--- find n1r in s26stage ---"
timeout 60 find /home/arm/junyang/s26stage -iname '*n1r*' 2>/dev/null || echo "FIND-S26-EMPTY"
echo "--- data/packs ---"
ls -la /home/arm/junyang/carbon/data/packs/ 2>&1 | head -40 || echo "DATAPACKS-MISSING"
echo "--- find n1r in data (maxdepth 3) ---"
timeout 60 find /home/arm/junyang/carbon/data -maxdepth 3 -iname '*n1r*' 2>/dev/null || echo "FIND-DATA-EMPTY"
echo "--- find n1r_int8 anywhere in carbon home (maxdepth 4) ---"
timeout 90 find /home/arm/junyang/carbon /home/arm/junyang/s26stage -maxdepth 4 -iname '*n1r_int8*' 2>/dev/null || echo "FIND-INT8-EMPTY"
echo "--- extract _OPTS + make_agent from any found pack file ---"
for f in $(timeout 60 find /home/arm/junyang/s26stage /home/arm/junyang/carbon/data -iname '*n1r*' -type f 2>/dev/null | head -10); do
  echo "FILE $f"
  grep -n -A 12 '_OPTS' "$f" | head -30
  grep -n -A 8 'def make_agent' "$f" | head -20
done
echo "--- smoke step-0 repo md5 ---"
git -C /home/arm/junyang/carbon-repo rev-parse HEAD
md5sum /home/arm/junyang/carbon-repo/code/rl2_ppo_rollout.py /home/arm/junyang/carbon-repo/code/eval_v53.py /home/arm/junyang/carbon-repo/code/cnet2_search.py /home/arm/junyang/carbon-repo/code/cnet2_np.py
echo "--- working-dir md5 (stale-copy check) ---"
md5sum /home/arm/junyang/carbon/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/eval_v53.py /home/arm/junyang/carbon/code/cnet2_search.py /home/arm/junyang/carbon/code/cnet2_np.py
echo "--- smoke_final_summary.json BEGIN ---"
cat /home/arm/junyang/carbon/data/smoke_pb_final/smoke_final_summary.json
echo "--- smoke_final_summary.json END ---"
echo "--- r12 key fields ---"
python3 - <<'PYEOF'
import json, os
p = "/home/arm/junyang/carbon/data/smoke_pb_final/r12_final_summary.json"
print("R12-SIZE", os.path.getsize(p))
try:
    d = json.load(open(p))
    if isinstance(d, dict):
        out = {k: v for k, v in d.items() if isinstance(v, (str, int, float, bool)) or v is None}
        print("R12-FINAL", json.dumps(out, ensure_ascii=False))
    else:
        print("R12-FINAL-TYPE", type(d).__name__, "len", len(d))
except Exception as e:
    print("R12-ERR", repr(e))
PYEOF
echo "DGXS26OPTS DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
