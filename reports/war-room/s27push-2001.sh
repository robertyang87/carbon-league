#!/bin/bash
# s27 push launch v2 (probe ssh now uses -n so it cannot eat the stdin-fed script).
# Runs on L20 via cloud-assistant; quick staging recheck on DGX + spot-27 host
# key scan/record + ssh probe, then launches the transfer worker in DGX
# background (log: ~/s27xfer.log). Marker: S27PUSH2001K
MARK=S27PUSH2001K
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S27PUSH2001K
ST=/home/arm/junyang/s26stage
TGT=47.117.49.49
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo '--- STAGE-QUICK-RECHECK ---'
du -sh "$ST"
echo -n "weights-files="; find "$ST/data/weights" -type f | wc -l
md5sum "$ST/code/exp/echo2.py" "$ST/code/exp/echo_e0.py" "$ST/code/eval_v53.py" "$ST/code/bigeval.py" "$ST/data/weights/cnet2_m3w5L.npz" "$ST/data/weights/cnet2_m3w3L.npz" "$ST/data/weights/cnet2_s64.npz"
echo '--- HOSTKEY-SCAN ---'
mkdir -p ~/.ssh; chmod 700 ~/.ssh
rm -f /tmp/s27hk.txt
timeout 30 ssh-keyscan -t ed25519,rsa "$TGT" > /tmp/s27hk.txt 2>/dev/null </dev/null
if [ -s /tmp/s27hk.txt ]; then
  ssh-keygen -lf /tmp/s27hk.txt
  grep -v -F "$TGT" ~/.ssh/known_hosts > /tmp/kh.txt 2>/dev/null || true
  cat /tmp/s27hk.txt >> /tmp/kh.txt
  mv /tmp/kh.txt ~/.ssh/known_hosts; chmod 600 ~/.ssh/known_hosts
  echo HOSTKEY-RECORDED
else
  echo KEYSCAN-FAILED
fi
echo '--- SSH-PROBE ---'
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'echo SSH-OK $(hostname)' 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK SSH-NOT-READY"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
echo '--- WRITE-WORKER ---'
cat > /home/arm/junyang/s27worker.sh <<'WKEOF'
#!/bin/bash
MARK=S27PUSH2001K
ST=/home/arm/junyang/s26stage
TGT=47.117.49.49
KEY="$1"
echo "$MARK WORKER-BEGIN $(date '+%F %T %Z')"
SSHCMD="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=20"
timeout -k 30 1500 rsync -a --timeout=180 -e "$SSHCMD" "$ST"/ root@"$TGT":/root/carbon/
RC=$?
echo "RSYNC-RC=$RC"
if [ $RC -ne 0 ]; then
  echo '--- TAR-FALLBACK ---'
  tar -C "$ST" -cf - . | timeout -k 30 1500 $SSHCMD root@"$TGT" 'mkdir -p /root/carbon && tar -C /root/carbon -xf -'
  echo "TAR-RC=$?"
fi
echo '--- TARGET-VERIFY ---'
timeout -k 30 240 $SSHCMD root@"$TGT" 'bash -s' <<'TGTEOF'
C=/root/carbon
echo "host: $(hostname)"
du -sh "$C"
echo -n "weights-files="; find "$C/data/weights" -type f | wc -l
md5sum "$C/code/exp/echo2.py" "$C/code/exp/echo_e0.py" "$C/code/eval_v53.py" "$C/code/bigeval.py" "$C/data/weights/cnet2_m3w5L.npz" "$C/data/weights/cnet2_m3w3L.npz" "$C/data/weights/cnet2_s64.npz"
ls -l "$C/.venv/bin/python"
"$C/.venv/bin/python" -c "import numpy; print('numpy', numpy.__version__)"
"$C/.venv/bin/python" -m py_compile "$C/code/bigeval.py" "$C/code/eval_v53.py" "$C/code/exp/echo2.py" "$C/code/exp/echo_e0.py" && echo PYCOMPILE-OK
TGTEOF
echo "TARGET-VERIFY-RC=$?"
echo "$MARK WORKER-DONE $(date '+%F %T %Z')"
WKEOF
chmod +x /home/arm/junyang/s27worker.sh
nohup bash /home/arm/junyang/s27worker.sh "$KEY" > /home/arm/junyang/s27xfer.log 2>&1 </dev/null &
echo "$MARK LAUNCHED pid=$!"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
