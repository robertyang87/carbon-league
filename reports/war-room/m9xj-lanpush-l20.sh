#!/bin/bash
# ds_m9xj_w DGX -> Studio over the home LAN (ZT data path broken ~2.6KB/s;
# ZT used here for small control commands only).
# Route A: DGX pushes directly to Studio mDNS/LAN address.
# Route B: Studio pulls from DGX LAN 10.188.0.23 (command issued via L20->Studio ssh).
# Marker: M9XJLAN2103C ; expected agg md5 9920540853a148f3e0252477bb5d9d51
MARK=M9XJLAN2103C
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '=== DGX part: discovery + route A + train progress ==='
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "dgx $(hostname) $(date '+%F %T %Z')"
echo '--- discovery ---'
cut -d' ' -f1 ~/.ssh/known_hosts 2>/dev/null | tr ',' '\n' | sort -u | head -20
getent hosts mac-studio.local || echo NO-MDNS-GETENT
ip neigh 2>/dev/null | head -12
ls ~/.ssh/
echo '--- train progress ---'
grep '"ep":' /home/arm/junyang/carbon/data/dq_logs/m9xj260r1.log | tail -2
grep '"ep":' /home/arm/junyang/carbon/data/dq_logs/m9xj260g1.log | tail -2
ps -eo pid,etime,args | grep cnet2_train | grep -v grep || echo NO-TRAIN-PROC
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- route A: DGX push to Studio ---'
for H in mac-studio.local; do
  timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=12 -o StrictHostKeyChecking=accept-new arm@$H 'echo ROUTEA-REACHABLE $(hostname)' 2>&1 | tail -1
  if tar cf - -C /home/arm/junyang/carbon/data ds_m9xj_w | timeout -k 20 300 ssh -o BatchMode=yes -o ConnectTimeout=12 arm@$H 'mkdir -p ~/junyang/carbon/data && tar xf - -C ~/junyang/carbon/data'; then
    echo "ROUTEA-PUSH-OK via $H"
  else
    echo "ROUTEA-PUSH-FAILED via $H"
  fi
done
echo '--- DGX part end ---'
DGXEOF
echo '=== Studio part: verify / route B pull via LAN ==='
timeout -k 30 600 ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' <<'STEOF'
echo "studio $(hostname) $(date '+%F %T %Z')"
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
AGG() { ~/junyang/carbon/.venv/bin/python /tmp/agg.py ~/junyang/carbon/data/ds_m9xj_w 2>/dev/null || echo NO-DIR; }
CUR=$(AGG); echo "STUDIO-CURRENT-AGG $CUR"
if [ "$CUR" = "9920540853a148f3e0252477bb5d9d51" ]; then
  echo "PRESENT-AND-MATCH (route A landed)"
else
  echo '--- route B: pull from DGX LAN 10.188.0.23 ---'
  timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 'echo PULL-REACHABLE $(hostname)' || echo PULL-SSH-FAILED
  mkdir -p ~/junyang/carbon/data
  rm -rf ~/junyang/carbon/data/ds_m9xj_w
  timeout -k 20 420 ssh -o BatchMode=yes -o ConnectTimeout=15 arm@10.188.0.23 'tar cf - -C /home/arm/junyang/carbon/data ds_m9xj_w' | tar xf - -C ~/junyang/carbon/data && echo ROUTEB-PULL-OK
  echo "STUDIO-FINAL-AGG $(AGG)"
fi
echo '--- studio ds listing ---'
ls ~/junyang/carbon/data/ds_m9xj_w 2>/dev/null | head -6
du -sh ~/junyang/carbon/data/ds_m9xj_w 2>/dev/null
STEOF
echo "$MARK expected 9920540853a148f3e0252477bb5d9d51"
echo "$MARK L20-END rc=$?"
