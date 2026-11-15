#!/bin/bash
echo "LAUNCH_BEGIN"
DATA=/Users/arm/junyang/carbon/data
if pgrep -f "cnet2_train.py" >/dev/null; then
  echo "ABORT: trainer already running"; pgrep -fl "cnet2_train.py"; echo "LAUNCH_END"; exit 0
fi
cat > $DATA/run_m3s2wr1.sh <<'LAEOF'
#!/bin/bash
cd /Users/arm/junyang/carbon/code || exit 1
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
export BC_INIT=/Users/arm/junyang/carbon/data/weights/cnet2_arch_n1_last.npz
/Users/arm/junyang/carbon/.venv/bin/python cnet2_train.py /Users/arm/junyang/carbon/data/ds_m3s2_w /Users/arm/junyang/carbon/data/weights/cnet2_m3s2wr1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}'
LAEOF
cat > $DATA/ship_m3s2wr1.sh <<'SHEOF'
#!/bin/bash
LOG=/Users/arm/junyang/carbon/data/m3s2wr1_ship.log
EP=/Users/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_ep19.npz
DST=arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz
for i in $(seq 1 90); do pgrep -f "cnet2_train.py" >/dev/null && break; sleep 10; done
echo "$(date '+%F %T') watcher: trainer seen, waiting for finish" >> "$LOG"
while pgrep -f "cnet2_train.py" >/dev/null; do sleep 60; done
sleep 15
if [ -f "$EP" ]; then
  LMD5=$(md5 -q "$EP")
  scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$EP" "$DST"
  RC=$?
  RMD5=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz" | awk '{print $1}')
  if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
  echo "$(date '+%F %T') shipped ep19 scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M" >> "$LOG"
else
  echo "$(date '+%F %T') trainer ended but EP19 MISSING" >> "$LOG"
fi
SHEOF
cd $DATA
nohup bash $DATA/run_m3s2wr1.sh > $DATA/m3s2wr1.log 2>&1 &
echo "launcher_pid=$!"
sleep 5
pgrep -fl "cnet2_train.py" || echo "TRAINER_NOT_SEEN_YET"
nohup bash $DATA/ship_m3s2wr1.sh > $DATA/m3s2wr1_ship.out 2>&1 &
echo "watcher_pid=$!"
sleep 2
pgrep -fl "ship_m3s2wr1" || echo "WATCHER_NOT_SEEN"
echo "LAUNCH_END"
