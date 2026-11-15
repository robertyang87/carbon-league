#!/bin/bash
MARK=WG1PROBE1907A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "WG1PROBE1907A STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,etime,command | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
ps -ax -o pid,etime,command | grep -E 'ship_' | grep -v grep || echo NO-SHIP-PROC
D=$HOME/junyang/carbon/data
echo '--- DS-DIRS ---'
ls -d $D/ds_m3s2_w $D/ds_m3s2 2>&1
du -sh $D/ds_m3s2_w 2>/dev/null
echo '--- DS-W-LAYOUT ---'
ls $D/ds_m3s2_w 2>/dev/null | head -20
echo '--- META ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os
for name in ("ds_m3s2_w", "ds_m3s2"):
    root = os.path.expanduser("~/junyang/carbon/data/" + name)
    mp = os.path.join(root, "meta.json")
    if not os.path.exists(mp):
        print(name, "NO-META"); continue
    d = json.load(open(mp))
    keep = {k: v for k, v in d.items() if k != "games"}
    print(name, json.dumps(keep)[:600])
    if "games" in d: print(name, "games:", len(d["games"]))
PYEOF
echo '--- INIT-AND-CODE ---'
ls -l $D/weights/cnet2_b128m2t3.npz $D/weights/cnet2_m3s2wr1_ep19.npz 2>&1
ls $HOME/junyang/carbon/code/cnet2_train.py
echo '--- DISK ---'
df -h / | tail -1
echo "WG1PROBE1907A STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
