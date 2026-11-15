#!/bin/bash
# Read-back probe: fetch PBSMOKE2352B's on-DGX summary JSONs (the run's own Actions log
# could not be claimed -- whole-log downloads kept dying). Prints compact extracts only.
# Marker: PBSUM0013A
MARK=PBSUM0013A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBSUM0013A DGX-BEGIN $(date '+%F %T %Z')"
hostname
ART=/home/arm/junyang/carbon/data/smoke_pb_2345
echo '--- ART LIST ---'
ls -la --time-style=full-iso $ART
echo '--- PS SMOKE ---'
ps aux | grep -E 'smoke_pb|r12b' | grep -v grep || echo "no smoke processes"
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader
echo '--- EXTRACT ---'
python3 - <<'PYEOF'
import json
art = "/home/arm/junyang/carbon/data/smoke_pb_2345"
try:
    s = json.load(open(art + "/smoke_summary2.json"))
    print("SMOKE2 weights:", s.get("weights"))
    print("SMOKE2 bot:", s.get("bot"))
    print("SMOKE2 summary:", json.dumps(s.get("summary")))
    for g in s.get("games", []):
        if g.get("term") or not g.get("len_ok") or not g.get("horizon_ok") \
           or (g.get("dw_err") or 0) > 0.01 or (g.get("dm_err") or 0) > 0.01:
            print("SMOKE2 FLAGGED-GAME", json.dumps(g))
    print("SMOKE2 n_games_listed:", len(s.get("games", [])), "errors:", s.get("errors"))
except Exception as e:
    print("SMOKE2 EXTRACT-ERROR", repr(e))
try:
    r = json.load(open(art + "/r12_summary2.json"))
    keep = {k: r.get(k) for k in ("chosen", "md5", "sig_mismatch_after_tta",
            "diff_stale_view_vs_raw", "diff_consumed_encfor_vs_raw", "error")}
    keep["n_tried"] = len(r.get("tried", []))
    print("R12-2", json.dumps(keep))
except Exception as e:
    print("R12-2 EXTRACT-ERROR", repr(e))
PYEOF
echo "PBSUM0013A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
