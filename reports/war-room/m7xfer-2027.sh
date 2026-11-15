#!/bin/bash
# Via L20 hop: locate cnet2_m7bsr1L.npz on DGX, md5-gate it, scp to h23, verify on h23.
# Marker: M7XFER2027E (claim only output containing this marker)
MARK=M7XFER2027E
echo "${MARK}_BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "M7XFER2027E DGX-BEGIN $(date '+%F %T %Z')"
F=$(find /home/arm/junyang -name 'cnet2_m7bsr1L.npz' 2>/dev/null | head -3)
echo "found: $F"
for f in $F; do md5sum "$f"; ls -la "$f"; done
SRC=$(echo "$F" | head -1)
M=$(md5sum "$SRC" | cut -d' ' -f1)
echo "src=$SRC md5=$M"
if [ "$M" != "0835830f792840ed66eef6d5be84b417" ]; then echo "ABORT DGX md5 mismatch"; echo "M7XFER2027E DGX-END"; exit 1; fi
timeout -k 30 300 scp -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 "$SRC" root@47.117.148.216:/root/carbon/data/weights/
echo "scp rc=$?"
echo "--- verify on h23 ---"
timeout -k 10 60 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@47.117.148.216 'md5sum /root/carbon/data/weights/cnet2_m7bsr1L.npz; ls -la /root/carbon/data/weights/cnet2_m7bsr1L.npz' 2>&1
echo "M7XFER2027E DGX-END"
DGXEOF
echo "${MARK}_END rc=$?"
