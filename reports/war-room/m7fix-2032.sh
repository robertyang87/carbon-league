#!/bin/bash
# Resolve cnet2_m7bsr1L.npz (target md5 0835830f792840ed66eef6d5be84b417) onto h23.
# Outer script runs on L20: md5-gate L20's copy, push it to DGX /tmp as fallback.
# On DGX: if a native candidate matches the md5 use it (Path A), else use the
# pushed copy (Path B); relay DGX->h23 and verify md5 on h23.
# Marker: M7FIX2032H (claim only output containing this marker)
MARK=M7FIX2032H
TARGET=0835830f792840ed66eef6d5be84b417
echo "${MARK}_BEGIN $(date '+%F %T %Z')"
LM=$(md5sum /root/carbon/data/weights/cnet2_m7bsr1L.npz | cut -d' ' -f1)
echo "L20 copy md5=$LM"
[ "$LM" = "$TARGET" ] || { echo "ABORT L20 copy md5 mismatch"; echo "${MARK}_END"; exit 1; }
timeout -k 30 120 scp -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=15 /root/carbon/data/weights/cnet2_m7bsr1L.npz arm@10.211.43.155:/tmp/cnet2_m7bsr1L.npz
echo "L20->DGX push rc=$?"
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<DGXEOF
echo "M7FIX2032H DGX-BEGIN \$(date '+%F %T %Z')"
W=/home/arm/junyang/carbon/data/weights
SRC=""
for f in \$W/cnet2_m7bsr1_ep19.npz \$W/cnet2_m7bsr1.npz \$W/cnet2_m7bsr1_last.npz; do
  [ -f "\$f" ] || continue
  m=\$(md5sum "\$f" | cut -d' ' -f1); echo "candidate \$f md5=\$m"
  if [ "\$m" = "$TARGET" ]; then SRC="\$f"; fi
done
if [ -n "\$SRC" ]; then
  echo "PATH_A src=\$SRC"
  cp "\$SRC" /tmp/cnet2_m7bsr1L.npz
else
  echo "PATH_B: using L20-pushed /tmp copy"
  m=\$(md5sum /tmp/cnet2_m7bsr1L.npz | cut -d' ' -f1); echo "tmp md5=\$m"
  [ "\$m" = "$TARGET" ] || { echo "ABORT tmp md5 mismatch"; echo "M7FIX2032H DGX-END"; exit 1; }
fi
timeout -k 30 300 scp -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 /tmp/cnet2_m7bsr1L.npz root@47.117.148.216:/root/carbon/data/weights/cnet2_m7bsr1L.npz
echo "relay scp rc=\$?"
echo "--- verify on h23 ---"
timeout -k 10 60 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_m7bsr1L.npz; ls -la /root/carbon/data/weights/cnet2_m7bsr1L.npz' 2>&1
echo "M7FIX2032H DGX-END"
DGXEOF
echo "${MARK}_END rc=$?"
