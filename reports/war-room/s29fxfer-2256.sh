#!/bin/bash
# H29: transfer calibration weights DGX -> spot-29 (Fable r1 required;
# M7Bs v2 pair + Carago r1 pre-staged for later jobs). Runs on L20,
# hops to DGX, DGX pushes directly to spot-29. Marker: S29FXFER2256
MARK=S29FXFER2256
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 900 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S29FXFER2256
TGT=47.117.170.99
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo "--- LOCATE + MD5 ON DGX ---"
declare -A SRC
for W in cnet2_fable_r1_fromstudio.npz cnet2_m7bsv2r1.npz cnet2_m7bsv2g1.npz cnet2_caragor1_ep19.npz; do
  P=""
  for D in /home/arm/junyang/carbon/data/weights /home/arm/junyang/s26stage/data/weights; do
    if [ -f "$D/$W" ]; then P="$D/$W"; break; fi
  done
  if [ -z "$P" ]; then
    P=$(timeout 60 find /home/arm/junyang -name "$W" 2>/dev/null | head -1)
  fi
  if [ -n "$P" ]; then
    echo "FOUND $W at $P md5=$(md5sum "$P" | cut -d' ' -f1) size=$(stat -c%s "$P")"
    SRC[$W]="$P"
  else
    echo "NOT-FOUND-ON-DGX $W"
  fi
done
if [ -z "${SRC[cnet2_fable_r1_fromstudio.npz]}" ]; then echo "$MARK ABORT fable source missing"; echo "$MARK DGX-END"; exit 0; fi
FM=$(md5sum "${SRC[cnet2_fable_r1_fromstudio.npz]}" | cut -d' ' -f1)
[ "$FM" = "8bec64d7618686f52f73d592e7099940" ] || { echo "$MARK ABORT fable md5 on DGX: $FM"; echo "$MARK DGX-END"; exit 0; }
echo "FABLE-SOURCE-MD5-OK"
echo "--- FIND WORKING KEY TO TARGET ---"
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'echo SSH-OK $(hostname)' 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK ABORT no working key to target"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
echo "--- TRANSFER ---"
for W in cnet2_fable_r1_fromstudio.npz cnet2_m7bsv2r1.npz cnet2_m7bsv2g1.npz cnet2_caragor1_ep19.npz; do
  P="${SRC[$W]}"
  [ -n "$P" ] || { echo "SKIP $W (not found)"; continue; }
  timeout -k 30 300 scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$P" root@"$TGT":/root/carbon/data/weights/"$W"
  echo "SCP $W rc=$?"
done
echo "--- VERIFY ON TARGET ---"
timeout -k 30 120 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'md5sum /root/carbon/data/weights/cnet2_fable_r1_fromstudio.npz /root/carbon/data/weights/cnet2_m7bsv2r1.npz /root/carbon/data/weights/cnet2_m7bsv2g1.npz /root/carbon/data/weights/cnet2_caragor1_ep19.npz 2>&1; ls -l /root/carbon/data/weights/ | grep -iE "fable|m7bsv2|carago"'
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
