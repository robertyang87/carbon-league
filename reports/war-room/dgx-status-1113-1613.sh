#!/bin/bash
# DGX status probe trip 2026-11-13 16:13 - READ-ONLY. Marker DGXSTATUS1613A-F.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Purpose: routine 15-min patrol. DGX expected idle standby (war-room 15:18 ruling);
# HEAD 5aa7b544 since 16:02; no pull order outstanding (re-pull is war-room's call).
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1613A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1613A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1613B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1613B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo TRACKING-REF:; git rev-parse origin/main 2>/dev/null; echo BEHIND-TRACKING:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---P1-MD5-REPO---; md5sum code/cnet2_build.py code/cnet2_train.py code/set_uw.py 2>/dev/null; echo ---P1-MD5-WORKDIR---; md5sum /home/arm/junyang/carbon/code/cnet2_build.py /home/arm/junyang/carbon/code/cnet2_train.py /home/arm/junyang/carbon/code/set_uw.py 2>/dev/null; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1613C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1613C; echo ---HAKIMI-WEIGHTS---; md5sum /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; echo ---HAKIMI-LOGS-TAIL---; tail -2 /home/arm/junyang/carbon/data/dq_logs/hakimi3r1.log; tail -2 /home/arm/junyang/carbon/data/dq_logs/hakimi3g1.log; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1613D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1613D; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---DS-META---; for d in ds_carago ds_fable ds_hakimi3; do echo -n "$d meta: "; python3 -c "import json,sys; m=json.load(open(\"/home/arm/junyang/carbon/data/$d/meta.json\")); print({k:m.get(k) for k in (\"n\",\"samples\",\"num_samples\",\"games\",\"num_games\") if k in m} or m)" 2>&1 | head -c 300; echo; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1613E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1613E; df -h / | tail -1; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1613-END $(date '+%H:%M:%S %Z')"
