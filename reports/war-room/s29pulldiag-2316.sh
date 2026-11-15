#!/bin/bash
# C1 diagnostic: single snapshot pull spot-29 via L20->DGX relay.
MARK=S29PD2316
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S29PD2316
TGT=47.117.170.99
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
date '+%F %T %Z'
echo "--- SUMMARIES RAW ---"
cat /root/carbon/data/bigeval.jsonl 2>/dev/null | cut -c1-300
echo "--- B64 fabler1_echo2 ---"
gzip -c /root/carbon/data/bigeval_live/fabler1_echo2.jsonl 2>/dev/null | base64 -w0; echo
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
