#!/bin/bash
# g1cal relay (runs on L20, hops to DGX): scp new eval_v53.py from DGX repo to h23,
# then md5-verify on h23. Fixes the CmdContent.ExceedLimit failure of the embed approach.
MARK="G1CALR7Q3"
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H23=47.117.148.216
echo "G1CALR7Q3 DGX-BEGIN $(date '+%F %T %Z')"
echo "--- repo eval_v53 md5 (source) ---"
md5sum /home/arm/junyang/carbon-repo/code/eval_v53.py
echo "--- refresh known_hosts for H23 ---"
ssh-keygen -f /home/arm/.ssh/known_hosts -R "$H23" >/dev/null 2>&1
ssh-keyscan -t ed25519 "$H23" 2>/dev/null >> /home/arm/.ssh/known_hosts
echo "--- scp eval_v53.py to H23 ---"
scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 /home/arm/junyang/carbon-repo/code/eval_v53.py root@"$H23":/root/carbon/code/eval_v53.py && echo SCP-OK || echo SCP-FAIL
echo "--- verify on H23 ---"
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$H23" 'md5sum /root/carbon/code/eval_v53.py; ls -la /root/carbon/code/eval_v53.py'
echo "G1CALR7Q3 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
