#!/bin/bash
# Small read-back probe for the final smoke SMKFIN0100A (war-room fallback, 01:28):
# read data/smoke_pb_final/ two summary JSONs + step-0 HEAD/md5 verification.
# Marker: SMKRB0128A
MARK=SMKRB0128A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "SMKRB0128A DGX-BEGIN $(date '+%F %T %Z')"
hostname
echo "--- ps smoke/play_batch ---"
ps aux | grep -E 'smoke_final|r12_final|play_batch' | grep -v grep || echo "NO-SMOKE-PROC"
echo "--- ls smoke_pb_final ---"
ls -l --time-style=full-iso /home/arm/junyang/carbon/data/smoke_pb_final/
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader || echo "SMI-FAIL"
echo "--- step0 HEAD ---"
git -C /home/arm/junyang/carbon-repo rev-parse HEAD
git -C /home/arm/junyang/carbon-repo branch --show-current
echo "--- step0 md5 code ---"
md5sum /home/arm/junyang/carbon/code/rl2_ppo_rollout.py /home/arm/junyang/carbon/code/eval_v53.py /home/arm/junyang/carbon/code/cnet2_search.py /home/arm/junyang/carbon/code/cnet2_np.py
echo "--- weights md5 ---"
md5sum /home/arm/junyang/carbon/data/weights/cnet2_b128m2t3.npz
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
echo "SMKRB0128A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
