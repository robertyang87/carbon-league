echo "M320LAUNCH2220A L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m320_launch.sh <<'STEOF'
echo "M320LAUNCH2220A STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
if pgrep -f "cnet2_train.py" >/dev/null; then echo "ABORT: trainer already running"; pgrep -fl "cnet2_train.py"; echo "M320LAUNCH2220A STUDIO_END"; exit 0; fi
echo TRAINER-FREE
[ -d $D/ds_m9xj320_w ] || { echo "ABORT: ds_m9xj320_w missing"; echo "M320LAUNCH2220A STUDIO_END"; exit 0; }
cd $HOME/junyang/carbon/code || exit 1
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m9xj320_w $D/weights/cnet2_m9xj320wr1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/m9xj320wr1.log 2>&1 < /dev/null &
echo $! > $D/m9xj320wr1.pid; echo "WR1_PID=$(cat $D/m9xj320wr1.pid)"
cat > $D/ship_m9xj320wr1.sh <<'SHEOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
LOG=$D/m9xj320wr1_ship.log
pid=$(cat $D/m9xj320wr1.pid 2>/dev/null)
if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
sleep 5
BEST=$($HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os
D = os.path.expanduser('~/junyang/carbon/data')
best_ep, best_val = 0, -1.0
try:
    for ln in open(os.path.join(D, 'm9xj320wr1.log'), errors='replace'):
        ln = ln.strip()
        if not ln.startswith('{'):
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if isinstance(r.get('val'), dict) and 'all' in r['val']:
            if r['val']['all'] > best_val:
                best_val, best_ep = r['val']['all'], int(r['ep'])
except FileNotFoundError:
    pass
print(best_ep, best_val)
PYEOF
)
BEST_EP=$(echo $BEST | awk '{print $1}')
BEST_VAL=$(echo $BEST | awk '{print $2}')
W=$D/weights
SRC=""
[ -n "$BEST_EP" ] && [ "$BEST_EP" != "0" ] && [ -f "$W/cnet2_m9xj320wr1_ep${BEST_EP}.npz" ] && SRC="$W/cnet2_m9xj320wr1_ep${BEST_EP}.npz"
[ -z "$SRC" ] && [ -f "$W/cnet2_m9xj320wr1.npz" ] && SRC="$W/cnet2_m9xj320wr1.npz"
MAIN_MD5=$([ -f "$W/cnet2_m9xj320wr1.npz" ] && md5 -q "$W/cnet2_m9xj320wr1.npz" || echo NA)
EP19_MD5=$([ -f "$W/cnet2_m9xj320wr1_ep19.npz" ] && md5 -q "$W/cnet2_m9xj320wr1_ep19.npz" || echo NA)
if [ -n "$SRC" ]; then
  LMD5=$(md5 -q "$SRC")
  scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$SRC" arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz < /dev/null
  RC=$?
  RMD5=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz" | awk '{print $1}')
  if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
  echo "$(date '+%F %T') shipped wr320 best_ep=$BEST_EP best_val=$BEST_VAL src=$(basename $SRC) scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M main_md5=$MAIN_MD5 ep19_md5=$EP19_MD5" >> "$LOG"
else
  echo "$(date '+%F %T') NO SOURCE FOUND wr320 best_ep=$BEST_EP best_val=$BEST_VAL main_md5=$MAIN_MD5 ep19_md5=$EP19_MD5" >> "$LOG"
fi
echo "$(date '+%F %T') WR320_SHIP_DONE" >> "$LOG"
SHEOF
nohup bash $D/ship_m9xj320wr1.sh > $D/m9xj320wr1_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
sleep 240
echo '--- THREE-PIECE ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
ps -o pid,etime,%cpu,rss,command -p $(cat $D/m9xj320wr1.pid) 2>&1 | cut -c1-200
echo '== wr320 log tail =='; tail -6 $D/m9xj320wr1.log
echo '== weights =='; ls -lt $D/weights/ | grep -i m9xj320wr1 | head -5
echo "M320LAUNCH2220A STUDIO_END"
STEOF
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m320_launch.sh
echo "launch_rc=$?"
echo "M320LAUNCH2220A L20_END"
