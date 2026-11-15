#!/bin/bash
# DGX status probe trip 14:13 follow-up H - read-only. Marker DGXSTATUS1413H.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "DGXSTATUS1413H $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413H; echo ---R1-LOGJSONL---; cat /home/arm/junyang/carbon/data/weights/cnet2_lite_v7r1_log.jsonl; echo ---G1-LOGJSONL---; cat /home/arm/junyang/carbon/data/weights/cnet2_lite_v7g1_log.jsonl; echo ---REPLICA-CHAIN---; ls -l /home/arm/junyang/carbon/data/dq_logs/replica_chain_lite_v7.log; tail -12 /home/arm/junyang/carbon/data/dq_logs/replica_chain_lite_v7.log; echo ---LITE-V8-MATCHES---; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon-repo/data/lite-v8_matches.json\")); print(\"lite-v8 matches:\", len(d))" 2>&1; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon-repo/data/krypton_matches.json\")); print(\"krypton matches:\", len(d))" 2>&1; echo ---END-H---' 2>&1
echo "rcH=$?"
echo "DGXSTATUS1413H-END $(date '+%H:%M:%S %Z')"
