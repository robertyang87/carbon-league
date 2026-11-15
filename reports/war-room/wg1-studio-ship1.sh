#!/bin/bash
MARK=WG1SHIP1930D
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "WG1SHIP1930D STUDIO-BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
echo '--- EP0-AND-BEST ---'
grep -h '"ep": 0' $D/m3s2wg1.log | head -2
grep -h '^best' $D/m3s2wg1.log | tail -1
echo '--- LOCAL-MD5 ---'
md5 -q $D/weights/cnet2_m3s2wg1_ep19.npz
echo '--- SHIP-LOG ---'
cat $D/m3s2wg1_ship.log 2>/dev/null || echo NO-SHIP-LOG
cat $D/m3s2wg1_ship.out 2>/dev/null | tail -3
echo '--- DS-VERIFY ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import numpy as np, json, os
root = os.path.expanduser("~/junyang/carbon/data/ds_m3s2_w")
meta = json.load(open(os.path.join(root, "meta.json")))
step = np.load(os.path.join(root, "step.npy"))
n = int(step.shape[0]); share = float((step >= 270).mean())
print("meta_total:", meta["total"], "rows:", n, "step>=270 share:", round(share, 4))
print("VERIFY:", "OK" if (n == 41488 and meta["total"] == 41488 and abs(share - 0.2437) < 0.005) else "MISMATCH")
PYEOF
echo '--- DGX-SIDE ---'
ssh -n -o BatchMode=yes -o ConnectTimeout=20 arm@10.188.0.23 'ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz 2>/dev/null; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz 2>/dev/null' < /dev/null || echo DGX-FILE-MISSING
echo "WG1SHIP1930D STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
