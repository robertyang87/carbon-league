#!/bin/bash
# DGX status probe 21b (follow-up): HEAD lag count, m7v2 fix log tail,
# ds_m7bs_v2 meta sample count, training re-check.
# Marker: DGXSTATUS1743B2K8 (claim only output containing this marker)
MARK=DGXSTATUS1743B2K8
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS1743B2K8 DGX-BEGIN $(date '+%F %T %Z')"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
echo '--- PROC-TRAIN ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|m7bsv2|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- HEAD-REMOTE-SHA (read-only ls-remote, count done via GitHub API from VM) ---'
(cd /home/arm/junyang/carbon-repo && git rev-parse HEAD 2>&1 && echo -n 'remote cnet2-search-v12: ' && timeout 60 git ls-remote origin cnet2-search-v12 2>&1)
echo '--- M7V2FIX-LOG-TAIL ---'
tail -15 /home/arm/junyang/carbon/data/dq_logs/m7v2fix.log 2>/dev/null
echo '--- BUILD-M7BSV2-LOG ---'
cat /home/arm/junyang/carbon/data/dq_logs/build_m7bsv2.log 2>/dev/null
echo '--- DS-M7BS-V2-META ---'
python3 -c "import json;m=json.load(open('/home/arm/junyang/carbon/data/ds_m7bs_v2/meta.json'));print({k:(len(v) if isinstance(v,list) else v) for k,v in m.items()})" 2>&1
echo '--- M7BSV2-LOGS-EXIST ---'
ls -l --time-style=+%F_%H:%M:%S /home/arm/junyang/carbon/data/dq_logs/m7bsv2r1.log /home/arm/junyang/carbon/data/dq_logs/m7bsv2g1.log 2>/dev/null || echo NO-M7BSV2-TRAIN-LOGS
echo "DGXSTATUS1743B2K8 DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
