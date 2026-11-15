#!/bin/bash
# s28 push launch (clone of s27push-2001.sh, which landed spot-27 20:06:06).
# Runs on L20 via cloud-assistant; quick staging recheck on DGX + spot-28 host
# key scan/record + ssh probe, then launches the transfer worker in DGX
# background (log: ~/s28xfer.log). Marker: S28PUSH2220
MARK=S28PUSH2220
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S28PUSH2220
ST=/home/arm/junyang/s26stage
TGT=47.116.17.244
FILES="code/exp/echo2.py code/exp/echo_e0.py code/eval_v53.py code/bigeval.py data/weights/cnet2_m3w5L.npz data/weights/cnet2_m3w3L.npz data/weights/cnet2_s64.npz data/weights/cnet2_m3s2r1L.npz data/weights/cnet2_m3s2g1L.npz data/weights/cnet2_k4v2r1_fromstudio.npz data/weights/cnet2_k4v2g1_fromstudio.npz"
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo '--- STAGE-QUICK-RECHECK ---'
du -sh "$ST"
echo -n "stage-files="; find "$ST" -type f | wc -l
echo -n "weights-files="; find "$ST/data/weights" -type f | wc -l
(cd "$ST" && md5sum $FILES > /tmp/s28src.md5)
cat /tmp/s28src.md5
echo '--- STAGE-EXPECTED-CHECK ---'
grep -q "^6d24da0eeeb28bad350d8b38f08ec3fc  code/eval_v53.py$" /tmp/s28src.md5 && echo EVAL53-OK || echo EVAL53-MISMATCH
grep -q "^a5822338d979abea807fbd2ab4f7dace  code/bigeval.py$" /tmp/s28src.md5 && echo BIGEVAL-OK || echo BIGEVAL-MISMATCH
grep -q "^15b6d754e6e21de33870074d3983a16f  code/exp/echo2.py$" /tmp/s28src.md5 && echo ECHO2-OK || echo ECHO2-MISMATCH
grep -q "^c969f7e7af94b3b8a2ea007044e80  code/exp/echo_e0.py$" /tmp/s28src.md5 && echo ECHOE0-OK || echo ECHOE0-MISMATCH
grep -q "^82fde46b2633950d3cbb29e8cacbfdc2  data/weights/cnet2_m3w5L.npz$" /tmp/s28src.md5 && echo M3W5-OK || echo M3W5-MISMATCH
grep -q "^e68d2660bab36bbee1ccd1a693cd67f0  data/weights/cnet2_m3w3L.npz$" /tmp/s28src.md5 && echo M3W3-OK || echo M3W3-MISMATCH
grep -q "^c84eef933d1d71272e89de3742081451  data/weights/cnet2_s64.npz$" /tmp/s28src.md5 && echo S64-OK || echo S64-MISMATCH
grep -q "^b6db08c9b076257c54817377bb93130d  data/weights/cnet2_m3s2r1L.npz$" /tmp/s28src.md5 && echo M3S2R1-OK || echo M3S2R1-MISMATCH
grep -q "^975c518c  data/weights/cnet2_m3s2g1L.npz$" /tmp/s28src.md5 && echo M3S2G1-PREFIX-OK || echo M3S2G1-PREFIX-MISMATCH
grep -q "^344cfde939debb0de7f8834af069ea16  data/weights/cnet2_k4v2r1_fromstudio.npz$" /tmp/s28src.md5 && echo K4R1-OK || echo K4R1-MISMATCH
grep -q "^a594efd6cdb865d978e511c66d299206  data/weights/cnet2_k4v2g1_fromstudio.npz$" /tmp/s28src.md5 && echo K4G1-OK || echo K4G1-MISMATCH
echo '--- HOSTKEY-SCAN ---'
mkdir -p ~/.ssh; chmod 700 ~/.ssh
rm -f /tmp/s28hk.txt
timeout 30 ssh-keyscan -t ed25519,rsa "$TGT" > /tmp/s28hk.txt 2>/dev/null </dev/null
if [ -s /tmp/s28hk.txt ]; then
  ssh-keygen -lf /tmp/s28hk.txt
  grep -v -F "$TGT" ~/.ssh/known_hosts > /tmp/kh.txt 2>/dev/null || true
  cat /tmp/s28hk.txt >> /tmp/kh.txt
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
cat > /home/arm/junyang/s28worker.sh <<'WKEOF'
#!/bin/bash
MARK=S28PUSH2220
ST=/home/arm/junyang/s26stage
TGT=47.116.17.244
KEY="$1"
FILES="code/exp/echo2.py code/exp/echo_e0.py code/eval_v53.py code/bigeval.py data/weights/cnet2_m3w5L.npz data/weights/cnet2_m3w3L.npz data/weights/cnet2_s64.npz data/weights/cnet2_m3s2r1L.npz data/weights/cnet2_m3s2g1L.npz data/weights/cnet2_k4v2r1_fromstudio.npz data/weights/cnet2_k4v2g1_fromstudio.npz"
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
FILES="code/exp/echo2.py code/exp/echo_e0.py code/eval_v53.py code/bigeval.py data/weights/cnet2_m3w5L.npz data/weights/cnet2_m3w3L.npz data/weights/cnet2_s64.npz data/weights/cnet2_m3s2r1L.npz data/weights/cnet2_m3s2g1L.npz data/weights/cnet2_k4v2r1_fromstudio.npz data/weights/cnet2_k4v2g1_fromstudio.npz"
echo "host: $(hostname)"
du -sh "$C"
echo -n "target-files="; find "$C" -type f | wc -l
echo -n "weights-files="; find "$C/data/weights" -type f | wc -l
(cd "$C" && md5sum $FILES > /tmp/s28tgt.md5)
cat /tmp/s28tgt.md5
ls -l "$C/.venv/bin/python"
"$C/.venv/bin/python" -c "import numpy; print('numpy', numpy.__version__)"
"$C/.venv/bin/python" -m py_compile "$C/code/bigeval.py" "$C/code/eval_v53.py" "$C/code/exp/echo2.py" "$C/code/exp/echo_e0.py" && echo PYCOMPILE-OK
TGTEOF
echo "TARGET-VERIFY-RC=$?"
echo '--- MD5-COMPARE (DGX staging vs target) ---'
timeout -k 30 60 scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT":/tmp/s28tgt.md5 /tmp/s28tgt.md5
if diff -q /tmp/s28src.md5 /tmp/s28tgt.md5 >/dev/null 2>&1; then
  echo MD5-ALL-MATCH
else
  echo MD5-MISMATCH
  diff /tmp/s28src.md5 /tmp/s28tgt.md5 || true
fi
echo "$MARK WORKER-DONE $(date '+%F %T %Z')"
WKEOF
chmod +x /home/arm/junyang/s28worker.sh
nohup bash /home/arm/junyang/s28worker.sh "$KEY" > /home/arm/junyang/s28xfer.log 2>&1 </dev/null &
echo "$MARK LAUNCHED pid=$!"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
