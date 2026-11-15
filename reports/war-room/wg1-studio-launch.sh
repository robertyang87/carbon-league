#!/bin/bash
MARK=WG1LAUNCH1910B
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "WG1LAUNCH1910B STUDIO-BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
echo '--- RECHECK-IDLE ---'
if pgrep -f "cnet2_train.py" >/dev/null; then echo "ABORT: trainer running"; pgrep -fl "cnet2_train.py"; echo "WG1LAUNCH1910B STUDIO-END"; exit 0; fi
echo TRAINER-FREE
echo '--- DS-VERIFY ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import numpy as np, json, os
root = os.path.expanduser("~/junyang/carbon/data/ds_m3s2_w")
meta = json.load(open(os.path.join(root, "meta.json")))
step = np.load(os.path.join(root, "step.npy"))
n = int(step.shape[0])
share = float((step >= 270).mean())
print("meta_total:", meta["total"], "rows:", n, "step>=270 share:", round(share, 4))
print("VERIFY:", "OK" if (n == 41488 and meta["total"] == 41488 and abs(share - 0.2437) < 0.005) else "MISMATCH")
PYEOF
echo '--- LAUNCH-WG1-CONTROL ---'
cd $HOME/junyang/carbon/code || exit 1
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m3s2_w $D/weights/cnet2_m3s2wg1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/m3s2wg1.log 2>&1 < /dev/null &
echo $! > $D/m3s2wg1.pid; echo "WG1_PID=$(cat $D/m3s2wg1.pid)"
cat > $D/ship_m3s2wg1.sh <<'SHEOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
LOG=$D/m3s2wg1_ship.log
pid=$(cat $D/m3s2wg1.pid 2>/dev/null)
if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
sleep 5
f=$D/weights/cnet2_m3s2wg1_ep19.npz
if [ -f "$f" ]; then
  LMD5=$(md5 -q "$f")
  scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$f" arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz
  RC=$?
  RMD5=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2wg1_fromstudio.npz" | awk '{print $1}')
  if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
  echo "$(date '+%F %T') shipped wg1 scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M" >> "$LOG"
else
  echo "$(date '+%F %T') EP19 MISSING wg1 $f" >> "$LOG"
fi
echo "$(date '+%F %T') WG1_SHIP_DONE" >> "$LOG"
SHEOF
nohup bash $D/ship_m3s2wg1.sh > $D/m3s2wg1_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
echo '--- SLEEP-240-FOR-EP0 ---'
sleep 240
echo '--- THREE-PIECE ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
ps -o pid,etime,%cpu,rss,command -p $(cat $D/m3s2wg1.pid) 2>&1 | cut -c1-200
echo '== wg1 log =='; tail -5 $D/m3s2wg1.log
echo '== weights =='; ls -lt $D/weights/ | grep -i m3s2wg1 | head -5
echo "WG1LAUNCH1910B STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "$MARK L20-END rc=$?"
