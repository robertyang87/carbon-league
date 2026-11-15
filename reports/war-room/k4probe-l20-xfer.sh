echo K4XFER2004_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGX-BEGIN $(date '+%F %T %Z')"
R1=$(find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -name 'cnet2_k4v2r1_fromstudio.npz' 2>/dev/null | head -1)
G1=$(find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -name 'cnet2_k4v2g1_fromstudio.npz' 2>/dev/null | head -1)
echo "R1=$R1"
echo "G1=$G1"
OK1=0; OK2=0
if [ -n "$R1" ]; then M1=$(md5sum "$R1" | cut -d' ' -f1); echo "R1-MD5 $M1"; case "$M1" in 344cfde9*) OK1=1; echo R1-MD5-OK;; *) echo R1-MD5-MISMATCH;; esac; else echo R1-NOT-FOUND; fi
if [ -n "$G1" ]; then M2=$(md5sum "$G1" | cut -d' ' -f1); echo "G1-MD5 $M2"; case "$M2" in a594efd6*) OK2=1; echo G1-MD5-OK;; *) echo G1-MD5-MISMATCH;; esac; else echo G1-NOT-FOUND; fi
if [ "$OK1" = 1 ] && [ "$OK2" = 1 ]; then
  timeout 120 scp -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=20 "$R1" "$G1" root@47.117.16.90:/root/carbon/data/weights/ && echo SCP-OK
  timeout -k 10 60 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=20 root@47.117.16.90 "md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz; ls -l /root/carbon/data/weights/cnet2_k4v2*"
else
  echo "TRANSFER-SKIPPED md5-or-missing"
fi
echo "DGX-END"
DGXEOF
echo "L20-END rc=$?"
echo K4XFER2004_END
