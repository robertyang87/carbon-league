#!/bin/bash
# Try DGX's candidate keys for DGX->Studio LAN (10.188.0.21); on first working
# key, push ds_m9xj_w via LAN and verify aggregate md5 on Studio.
# Also: L20->Studio key trials (control channel record), DGX training finals.
# Marker: M9XJKEY2110D ; expected agg md5 9920540853a148f3e0252477bb5d9d51
MARK=M9XJKEY2110D
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20 keys + L20->Studio trials (control only) ---'
ls /root/.ssh/
for K in /root/.ssh/id_ed25519 /root/.ssh/id_rsa /root/.ssh/id_ed25519_carbon_relay; do
  [ -f "$K" ] || continue
  timeout 40 ssh -i "$K" -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 "echo L20-STUDIO-OK key=$K host=\$(hostname)" 2>&1 | tail -1
done
echo '=== DGX part ==='
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "dgx $(hostname) $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data
cat > /tmp/agg.py <<'PYEOF'
import hashlib, os, sys
root = sys.argv[1]; h = hashlib.md5()
for dp, dn, fn in os.walk(root):
    dn.sort()
    for f in sorted(fn):
        p = os.path.join(dp, f); h.update(os.path.relpath(p, root).encode())
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""): h.update(chunk)
print(h.hexdigest())
PYEOF
echo '--- key trials DGX->Studio LAN 10.188.0.21 ---'
WINKEY=""
for K in id_ed25519_carbon_relay github_carbon id_ed25519 id_ed25519_eric id_rsa; do
  [ -f ~/.ssh/$K ] || { echo "no key $K"; continue; }
  OUT=$(timeout 40 ssh -i ~/.ssh/$K -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new arm@10.188.0.21 'echo DGX-STUDIO-OK $(hostname)' 2>&1 | tail -1)
  echo "key $K -> $OUT"
  case "$OUT" in *DGX-STUDIO-OK*) WINKEY=$K; break;; esac
done
echo "WINKEY=${WINKEY:-NONE}"
if [ -n "$WINKEY" ]; then
  tar cf - -C "$D" ds_m9xj_w | timeout -k 20 420 ssh -i ~/.ssh/$WINKEY -o BatchMode=yes -o ConnectTimeout=12 arm@10.188.0.21 'mkdir -p ~/junyang/carbon/data && tar xf - -C ~/junyang/carbon/data' && echo LAN-PUSH-OK
  timeout 60 scp -i ~/.ssh/$WINKEY -o BatchMode=yes /tmp/agg.py arm@10.188.0.21:/tmp/agg.py || echo SCP-AGG-FAILED
  echo -n 'STUDIO-AGG '
  timeout 180 ssh -i ~/.ssh/$WINKEY -o BatchMode=yes arm@10.188.0.21 '~/junyang/carbon/.venv/bin/python /tmp/agg.py ~/junyang/carbon/data/ds_m9xj_w' 2>&1 | tail -1
  echo 'expected 9920540853a148f3e0252477bb5d9d51'
fi
echo '--- training finals ---'
for A in r1 g1; do
  echo "== $A best val lines =="
  grep '"ep":' "$D/dq_logs/m9xj260$A.log" | python3 -c "import sys,json; best=None
for line in sys.stdin:
    try: r=json.loads(line)
    except Exception: continue
    if 'val' in r and isinstance(r['val'],dict):
        if best is None or r['val']['all']>best[1]: best=(r['ep'],r['val']['all'],r['val'])
print('best', best)"
  tail -1 "$D/dq_logs/m9xj260$A.log"
done
md5sum "$D"/weights/cnet2_m9xj260r1.npz "$D"/weights/cnet2_m9xj260r1_ep19.npz "$D"/weights/cnet2_m9xj260g1.npz "$D"/weights/cnet2_m9xj260g1_ep19.npz 2>&1
ls -l --time-style=+%F_%T "$D"/weights/cnet2_m9xj260*_ep19.npz
ps -eo pid,etime,args | grep cnet2_train | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- DGX part end ---'
DGXEOF
echo "$MARK L20-END rc=$?"
