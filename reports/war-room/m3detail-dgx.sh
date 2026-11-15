#!/bin/bash
# L20-side wrapper: ssh to DGX, read m3s2r1 training log(s) in dq_logs (READ ONLY).
MARK="M3DETAILK8X2"
echo "MARKER $MARK START"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK="M3DETAILK8X2"
echo "MARKER $MARK DGX-START"
D=/home/arm/junyang/carbon/data/dq_logs
echo "--- LS DQ_LOGS M3S2R1 ---"
ls -la $D | grep -i m3s2r1 || echo "no m3s2r1 files in dq_logs"
for f in $D/*m3s2r1*; do
  [ -f "$f" ] || continue
  echo "=== FILE $f ==="
  wc -c "$f"
  echo "--- HEAD12 ---"
  head -12 "$f"
  echo "--- VAL LINES ---"
  grep -in "val" "$f"
  echo "--- BEST LINES ---"
  grep -in "best" "$f"
  echo "--- TAIL3 ---"
  tail -3 "$f"
done
echo "MARKER $MARK DGX-END"
DGXEOF
echo "MARKER $MARK L20-END rc=$?"
