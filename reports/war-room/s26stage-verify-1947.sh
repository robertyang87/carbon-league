#!/bin/bash
# s26stage verify 1947: read-only check of DGX staging copy /home/arm/junyang/s26stage
# for spot-27 re-push. No writes, no push. Marker: S26STAGE1947QZ
MARK=S26STAGE1947QZ
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
S=/home/arm/junyang/s26stage
echo "S26STAGE1947QZ DGX-BEGIN $(date '+%F %T %Z')"
echo '--- SIZE ---'
du -sh "$S"
echo -n "total-files="; find "$S" -type f | wc -l
echo '--- WEIGHTS ---'
echo -n "weights-all-files="; find "$S/data/weights" -type f 2>/dev/null | wc -l
echo -n "weights-npz-files="; ls "$S"/data/weights/*.npz 2>/dev/null | wc -l
du -sh "$S/data/weights"
echo '--- MD5 ---'
for f in exp/echo2.py exp/echo_e0.py code/eval_v53.py code/bigeval.py data/weights/cnet2_m3w5L.npz data/weights/cnet2_m3w3L.npz data/weights/cnet2_s64.npz; do
  if [ -f "$S/$f" ]; then echo -n "$f: "; md5sum "$S/$f" | awk '{print $1}'; else echo "$f: MISSING"; fi
done
echo '--- VENV ---'
ls -l "$S/.venv/bin/python" 2>&1
"$S/.venv/bin/python" -c "import numpy; print('numpy', numpy.__version__)" 2>&1
echo '--- PYCOMPILE ---'
"$S/.venv/bin/python" -m py_compile "$S/code/bigeval.py" "$S/code/eval_v53.py" "$S/exp/echo2.py" "$S/exp/echo_e0.py" && echo PYCOMPILE-OK
echo "S26STAGE1947QZ DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
