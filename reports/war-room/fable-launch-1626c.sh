set -x
echo "== FABLELAUNCH-1626D5 BEGIN =="
date '+%F %T %Z'
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
set -x
echo "STUDIO-START $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
echo "-- stale check --"
pgrep -af cnet2_train || echo NO_TRAIN_PROC
ls -ld $D/ds_fable 2>&1
echo "-- preflight DGX ds_fable --"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX 'du -sh /home/arm/junyang/carbon/data/ds_fable; ls /home/arm/junyang/carbon/data/ds_fable; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/ds_fable/meta.json\"));print(\"games\",len(d[\"games\"]),\"total\",d[\"total\"],\"teachers\",d[\"teachers\"])"; find /home/arm/junyang/carbon/data/ds_fable -type f | wc -l'
echo "-- rsync ds_fable DGX-LAN->Studio --"
rsync -avz --partial -e "ssh -n -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon/data/ds_fable/ $D/ds_fable/ < /dev/null
echo "RSYNC_RC=$?"
echo "-- verify local ds_fable --"
du -sh $D/ds_fable
ls -la $D/ds_fable
python3 -c "import json;d=json.load(open(\"$D/ds_fable/meta.json\"));print(\"games\",len(d[\"games\"]),\"total\",d[\"total\"],\"teachers\",d[\"teachers\"])"
find $D/ds_fable -type f | wc -l
echo "-- aggregate md5 DGX --"
ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX "cd /home/arm/junyang/carbon/data/ds_fable && find . -type f | sort | xargs md5sum | awk '{print \$1, \$2}' | md5sum" < /dev/null
echo "-- aggregate md5 Studio --"
cd $D/ds_fable && find . -type f | sort | xargs md5 -r | awk '{print $1, $2}' | md5
cd $HOME/junyang/carbon
echo "-- weights preflight --"
ls -l $D/weights/cnet2_arch_n1_last.npz $D/weights/cnet2_b128m2t3.npz 2>&1
for f in cnet2_arch_n1_last.npz cnet2_b128m2t3.npz; do
  if [ ! -f $D/weights/$f ]; then
    echo "MISSING $f, fetching from DGX"
    rsync -avz --partial -e "ssh -n -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon/data/weights/$f $D/weights/$f < /dev/null
  fi
done
echo "-- code preflight --"
ls -l $HOME/junyang/carbon/code/cnet2_train.py 2>&1
if [ ! -f $HOME/junyang/carbon/code/cnet2_train.py ]; then
  echo "MISSING cnet2_train.py, fetching code from DGX"
  rsync -avz --partial -e "ssh -n -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon/code/ $HOME/junyang/carbon/code/ < /dev/null
fi
echo "-- launch dual-arm training --"
cd $HOME/junyang/carbon/code
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
rm -f $D/fable_r1.log $D/fable_g1.log
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_fable $D/weights/cnet2_fabler1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/fable_r1.log 2>&1 < /dev/null &
echo $! > /tmp/fable_r1.pid
echo "R1_PID=$(cat /tmp/fable_r1.pid)"
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_fable $D/weights/cnet2_fableg1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/fable_g1.log 2>&1 < /dev/null &
echo $! > /tmp/fable_g1.pid
echo "G1_PID=$(cat /tmp/fable_g1.pid)"
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
    rsync -avz -e "ssh -n -o BatchMode=yes -o ConnectTimeout=20" "$f" $DGX:/home/arm/junyang/carbon/data/weights/cnet2_fable_${arm}_fromstudio.npz < /dev/null
    echo "SHIPPED_${arm} rc=$? $(date '+%F %T')"
  else
    echo "MISSING_EP19_${arm} $f"
  fi
done
echo "FABLE_SHIP_DONE $(date '+%F %T')"
FABEOF
nohup bash /tmp/fable_ship.sh > /tmp/fable_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
echo "-- sleep 150 for ep0 --"
sleep 150
echo "-- three-piece verify $(date '+%F %T %Z') --"
echo "PIDS: r1=$(cat /tmp/fable_r1.pid 2>/dev/null) g1=$(cat /tmp/fable_g1.pid 2>/dev/null)"
pgrep -af cnet2_train || echo NO_TRAIN_PROC
ps -o pid,etime,%cpu,rss,command -p $(cat /tmp/fable_r1.pid 2>/dev/null),$(cat /tmp/fable_g1.pid 2>/dev/null) 2>&1
echo "-- device lines --"
grep -iE 'mps|device|cuda' $D/fable_r1.log | head -5
grep -iE 'mps|device|cuda' $D/fable_g1.log | head -5
echo "-- r1 log tail --"
tail -6 $D/fable_r1.log
echo "-- g1 log tail --"
tail -6 $D/fable_g1.log
echo "-- ep0 lines --"
grep -E 'ep0|epoch 0|val' $D/fable_r1.log | tail -4
grep -E 'ep0|epoch 0|val' $D/fable_g1.log | tail -4
echo "-- ep files so far --"
ls -lt $D/weights/ | grep -i fable | head -10
echo "STUDIO-END $(date '+%F %T %Z')"
STEOF
echo "SSH_RC=$?"
echo "== FABLELAUNCH-1626D5 END =="
