echo "K4STULCH1 STUDIO_BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
DGX=arm@10.188.0.23
echo '--- WAIT-WR1-FREE (max 120s) ---'
for i in $(seq 1 12); do
  if ! pgrep -f "cnet2_train.py" >/dev/null; then echo "trainer-free at check $i"; break; fi
  echo "trainer still running, wait 10s ($i)"; sleep 10
done
if pgrep -f "cnet2_train.py" >/dev/null; then
  echo "ABORT: trainer still running"; pgrep -fl "cnet2_train.py"; echo "K4STULCH1 STUDIO_END"; exit 0
fi
echo '--- WR1-FINAL ---'
grep -h '"ep": 19' $D/m3s2wr1.log | tail -1
grep -h '^best' $D/m3s2wr1.log | tail -1
tail -2 $D/m3s2wr1_ship.log 2>/dev/null
ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX 'ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m3s2wr1_fromstudio.npz' < /dev/null
md5 -q $D/weights/cnet2_m3s2wr1_ep19.npz
echo '--- SYNC-DS-K4V2 (tar over LAN ssh, no rsync) ---'
if [ -d $D/ds_k4v2 ]; then echo "ABORT: ds_k4v2 already exists on Studio"; echo "K4STULCH1 STUDIO_END"; exit 0; fi
ssh -n -o BatchMode=yes -o ConnectTimeout=20 $DGX 'cd /home/arm/junyang/carbon/data && tar cf - ds_k4v2' < /dev/null | tar xf - -C $D/
echo "TAR_PIPE_RC=$?"
du -sh $D/ds_k4v2; find $D/ds_k4v2 -type f | wc -l
echo '--- DS-VERIFY ---'
$HOME/junyang/carbon/.venv/bin/python - <<'PYEOF'
import json, os, hashlib
root = os.path.expanduser("~/junyang/carbon/data/ds_k4v2")
d = json.load(open(os.path.join(root, "meta.json")))
print("meta games:", len(d["games"]), "total:", d["total"], "teachers:", d["teachers"])
lines = []
for dp, dn, fn in os.walk(root):
    for f in fn:
        p = os.path.join(dp, f)
        rel = "./" + os.path.relpath(p, root)
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        lines.append((rel, h.hexdigest()))
lines.sort()
agg = hashlib.md5("".join(f"{h}  {rel}\n" for rel, h in lines).encode()).hexdigest()
print("files:", len(lines), "aggregate_md5:", agg)
print("MATCH_DGX:", agg == "b1ce3cad95dedfc91386b6b425f393f2")
PYEOF
echo '--- INIT-WEIGHTS ---'
ls -l $D/weights/cnet2_arch_n1_last.npz $D/weights/cnet2_b128m2t3.npz
echo '--- LAUNCH-DUAL-ARM ---'
cd $HOME/junyang/carbon/code || exit 1
export BC_DEVICE=mps BC_RAM=1 BC_THREADS=8 BC_USE_W=1 BC_USE_UW=1 BC_BS=256 BC_SAVE_EPOCHS=1 BC_PCT=0.1 BC_LR=3e-4
BC_INIT=$D/weights/cnet2_arch_n1_last.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_k4v2 $D/weights/cnet2_k4v2r1.npz 20 '{"C":128,"NB":8,"D":128,"ZI":1,"TA":4,"TNH":4}' > $D/k4v2r1.log 2>&1 < /dev/null &
echo $! > $D/k4v2r1.pid; echo "R1_PID=$(cat $D/k4v2r1.pid)"
BC_INIT=$D/weights/cnet2_b128m2t3.npz nohup $HOME/junyang/carbon/.venv/bin/python cnet2_train.py $D/ds_k4v2 $D/weights/cnet2_k4v2g1.npz 20 '{"C":128,"NB":8,"D":128}' > $D/k4v2g1.log 2>&1 < /dev/null &
echo $! > $D/k4v2g1.pid; echo "G1_PID=$(cat $D/k4v2g1.pid)"
cat > $D/ship_k4v2.sh <<'SHEOF'
#!/bin/bash
D=$HOME/junyang/carbon/data
LOG=$D/k4v2_ship.log
for arm in r1 g1; do
  pid=$(cat $D/k4v2${arm}.pid 2>/dev/null)
  if [ -n "$pid" ]; then while kill -0 "$pid" 2>/dev/null; do sleep 30; done; fi
  sleep 5
  f=$D/weights/cnet2_k4v2${arm}_ep19.npz
  if [ -f "$f" ]; then
    LMD5=$(md5 -q "$f")
    scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new "$f" arm@10.188.0.23:/home/arm/junyang/carbon/data/weights/cnet2_k4v2${arm}_fromstudio.npz
    RC=$?
    RMD5=$(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.188.0.23 "md5sum /home/arm/junyang/carbon/data/weights/cnet2_k4v2${arm}_fromstudio.npz" | awk '{print $1}')
    if [ "$LMD5" = "$RMD5" ]; then M=YES; else M=NO; fi
    echo "$(date '+%F %T') shipped ${arm} scp_rc=$RC local_md5=$LMD5 dgx_md5=$RMD5 match=$M" >> "$LOG"
  else
    echo "$(date '+%F %T') EP19 MISSING ${arm} $f" >> "$LOG"
  fi
done
echo "$(date '+%F %T') K4V2_SHIP_DONE" >> "$LOG"
SHEOF
nohup bash $D/ship_k4v2.sh > $D/k4v2_ship.out 2>&1 < /dev/null &
echo "SHIP_WATCHER_PID=$!"
echo '--- SLEEP-180-FOR-EP0 ---'
sleep 180
echo '--- THREE-PIECE ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
ps -o pid,etime,%cpu,rss,command -p $(cat $D/k4v2r1.pid),$(cat $D/k4v2g1.pid) 2>&1 | cut -c1-200
echo '== r1 log =='; tail -4 $D/k4v2r1.log
echo '== g1 log =='; tail -4 $D/k4v2g1.log
echo '== weights =='; ls -lt $D/weights/ | grep -i k4v2 | head -8
echo "K4STULCH1 STUDIO_END $(date '+%F %T %Z')"
