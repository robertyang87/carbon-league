#!/bin/bash
# DGX status probe trip 14:43 follow-up H2 - read-only. Marker DGXSTATUS1443H2.
# Remote shell is zsh: avoid ? and unquoted globs in echo text.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"
echo "DGXSTATUS1443H2 $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443H2; cd /home/arm/junyang/carbon-repo; echo ANCESTOR-CHECK; git merge-base --is-ancestor 45d57c1c HEAD && echo YES-ANCESTOR || echo NO-NOT-ANCESTOR; echo BEHIND-COMMITS; git log --oneline HEAD..origin/main | head -8; echo HAKIMI3-WEIGHTS; ls -lt /home/arm/junyang/carbon/data/weights/ | grep -i hakimi | head -6; echo DS-HAKIMI3-META; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon/data/ds_hakimi3/meta.json\")); print(d)" 2>&1; echo HAKIMI3-LOG; tail -3 /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1_log.jsonl 2>&1; echo PS-GPU; ps -eo pid,etime,args | grep cnet2_train | grep -v grep; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo END-H2' 2>&1
echo "rcH2=$?"
