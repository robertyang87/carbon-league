#!/bin/bash
# C1 one-shot: read Fable g1 chain status + summaries from spot-29.
MARK="S29FG1DIAG2346"
echo "${MARK}_BEGIN"
date '+%F %T %Z'
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S29FG1DIAG2346
TGT=47.117.170.99
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN"
cat /root/carbon/data/fableg1_cal.status 2>/dev/null || echo NO-STATUS
echo "--- COUNTS ---"
wc -l /root/carbon/data/bigeval_live/fableg1_m2.jsonl /root/carbon/data/bigeval_live/fableg1_echo2.jsonl 2>/dev/null
echo "bigeval-procs: $(pgrep -fc bigeval.py || echo 0)"
echo "--- SUMMARIES ---"
grep -h '"tag": "fableg1' /root/carbon/data/bigeval.jsonl 2>/dev/null | cut -c1-400
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
