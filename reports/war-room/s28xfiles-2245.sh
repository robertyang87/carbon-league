#!/bin/bash
# s28 +3 files probe: list target files newer than the rsync completion
# (22:36:36) to identify the 3 files beyond 6039 staging + 2 K4.
# Runs on L20 via cloud-assistant. Marker: S28XFILES2245
MARK=S28XFILES2245
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S28XFILES2245
TGT=47.116.17.244
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo '--- STAGING-COUNT-NOW ---'
echo -n "stage-files="; find /home/arm/junyang/s26stage -type f | wc -l
echo -n "stage-weights="; find /home/arm/junyang/s26stage/data/weights -type f | wc -l
timeout -k 30 120 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
C=/root/carbon
echo '--- TARGET-NEWER-THAN-RSYNC-DONE ---'
find "$C" -type f -newermt "2026-11-12 22:36:36" -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort
echo '--- TARGET-TOPLEVEL ---'
ls -la "$C"
echo -n "target-files="; find "$C" -type f | wc -l
echo -n "weights-files="; find "$C/data/weights" -type f | wc -l
TGTEOF
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
