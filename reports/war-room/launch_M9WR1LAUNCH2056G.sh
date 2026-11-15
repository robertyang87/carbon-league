echo "M9WR1LAUNCH2056G L20_BEGIN $(date '+%H:%M:%S')"
cat > /tmp/m9wr1_launch.sh <<'STEOF'
echo "M9WR1LAUNCH2056G STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
if pgrep -f "cnet2_train.py" >/dev/null; then echo "ABORT: trainer already running"; pgrep -fl "cnet2_train.py"; echo "M9WR1LAUNCH2056G STUDIO_END"; exit 0; fi
echo TRAINER-FREE
cd $HOME/junyang/carbon/code || exit 1
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_m9xj_w $D/weights/cnet2_m9xj260wr1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/m9xj260wr1.log 2>&1 < /dev/null &
echo $! > $D/m9xj260wr1.pid; echo "WR1_PID=$(cat $D/m9xj260wr1.pid)"
cat > $D/ship_m9xj260wr1.sh <<'SHEOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
LOG=$D/m9xj260wr1_ship.log
pid=$(cat $D/m9xj260wr1.pid 2>/dev/null)
if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
sleep 5
f=$D/weights/cnet2_m9xj260wr1_ep19.npz
if [ -f "$f" ]; then
  LMD5=$(md5 -q "$f")
  scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$f" arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_m9xj260wr1_fromstudio.npz < /dev/null
  RC=$?
  RMD5=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj260wr1_fromstudio.npz" | awk '{print $1}')
  if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
  echo "$(date '+%F %T') shipped wr1 scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M" >> "$LOG"
else
  echo "$(date '+%F %T') EP19 MISSING wr1 $f" >> "$LOG"
fi
echo "$(date '+%F %T') WR1_SHIP_DONE" >> "$LOG"
SHEOF
nohup bash $D/ship_m9xj260wr1.sh > $D/m9xj260wr1_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
sleep 240
echo '--- THREE-PIECE ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
ps -o pid,etime,%cpu,rss,command -p $(cat $D/m9xj260wr1.pid) 2>&1 | cut -c1-200
echo '== wr1 log tail =='; tail -6 $D/m9xj260wr1.log
echo '== weights =='; ls -lt $D/weights/ | grep -i m9xj260wr1 | head -5
echo "M9WR1LAUNCH2056G STUDIO_END"
STEOF
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.68 'bash -s' < /tmp/m9wr1_launch.sh
echo "launch_rc=$?"
echo "M9WR1LAUNCH2056G L20_END"
