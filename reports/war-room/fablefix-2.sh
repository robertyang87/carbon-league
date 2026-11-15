echo "== FABLEFIXQ7B BEGIN =="
date '+%F %T %Z'
timeout 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
D=$HOME/junyang/carbon/data
C=$HOME/junyang/carbon/code
DGX=arm@10.188.0.23
fail(){ echo "FABLEFIXQ7B FAIL: $1"; exit 1; }
echo "STUDIO $(hostname) $(date '+%F %T %Z')"
$HOME/junyang/carbon/.venv/bin/python -c "import json;d=json.load(open('$D/ds_fable/meta.json'));assert len(d['games'])==181 and d['total']==54119, 'meta mismatch';print('META_OK')" || fail "dataset re-verify failed"
[ -f "$D/ds_fable/maps.npy" ] || fail "maps.npy missing"
if pgrep -f cnet2_train >/dev/null; then echo "EXISTING_TRAIN_PROC:"; pgrep -af cnet2_train; fail "a cnet2_train process already exists - not touching it"; fi
echo "NO_TRAIN_PROC confirmed"
for f in cnet2_arch_n1_last.npz cnet2_b128m2t3.npz; do
  if [ ! -f "$D/weights/$f" ]; then
    echo "fetching $f via scp"
    scp -o BatchMode=yes -o ConnectTimeout=20 $DGX:/home/arm/junyang/carbon/data/weights/$f "$D/weights/$f" < /dev/null || fail "scp $f failed"
  fi
  ls -l "$D/weights/$f"
done
[ -f "$C/cnet2_train.py" ] || fail "cnet2_train.py missing"
cd "$C" || fail "cd code dir"
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
rm -f "$D/fable_r1.log" "$D/fable_g1.log"
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_fable $D/weights/cnet2_fabler1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/fable_r1.log 2>&1 < /dev/null &
echo $! > /tmp/fable_r1.pid
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_fable $D/weights/cnet2_fableg1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/fable_g1.log 2>&1 < /dev/null &
echo $! > /tmp/fable_g1.pid
echo "R1_PID=$(cat /tmp/fable_r1.pid) G1_PID=$(cat /tmp/fable_g1.pid)"
cat > /tmp/fable_ship.sh <<'FABEOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
for arm in r1 g1; do
  pid=$(cat /tmp/fable_${arm}.pid 2>/dev/null)
  if [ -n "$pid" ]; then
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  fi
  sleep 5
  f=$D/weights/cnet2_fable${arm}_ep19.npz
  if [ -f "$f" ]; then
    scp -o BatchMode=yes -o ConnectTimeout=20 "$f" $DGX:/home/arm/junyang/carbon/data/weights/cnet2_fable_${arm}_fromstudio.npz
    echo "SHIPPED_${arm} rc=$? $(date '+%F %T')"
    L=$(md5 -q "$f")
    R=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "md5sum /home/arm/junyang/carbon/data/weights/cnet2_fable_${arm}_fromstudio.npz | awk '{print \$1}'")
    echo "SHIP_MD5_${arm} local=$L remote=$R"
  else
    echo "MISSING_EP19_${arm} $f"
  fi
done
echo "FABLE_SHIP_DONE $(date '+%F %T')"
FABEOF
nohup bash /tmp/fable_ship.sh > /tmp/fable_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
echo "-- sleep 120 for ep0 --"
sleep 120
echo "-- three-piece verify $(date '+%F %T %Z') --"
pgrep -af cnet2_train || echo NO_TRAIN_PROC
ps -o pid,etime,%cpu,rss,command -p $(cat /tmp/fable_r1.pid),$(cat /tmp/fable_g1.pid) 2>&1
echo "LOG_MTIME r1=$(stat -f %Sm $D/fable_r1.log) g1=$(stat -f %Sm $D/fable_g1.log)"
echo "-- device lines --"
grep -iE 'device|mps' $D/fable_r1.log | head -3
grep -iE 'device|mps' $D/fable_g1.log | head -3
echo "-- r1 log tail --"
tail -5 $D/fable_r1.log
echo "-- g1 log tail --"
tail -5 $D/fable_g1.log
echo "-- ep files so far --"
ls -lt $D/weights/ | grep -i fable | head -6
echo "FABLEFIXQ7B DONE"
STEOF
echo "SSH_RC=$?"
echo "== FABLEFIXQ7B END =="
