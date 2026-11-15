#!/bin/bash
# DGX status probe trip 13:58 - read-only. Marker DGXSTATUS1358A-E.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50).
# Purpose: verify Lite-V7 dual-arm training release (war-room 13:56) - has index/dataset build or training started?
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1358A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1358B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358B; cd /home/arm/junyang/carbon-repo && timeout 60 git fetch origin 2>&1 | tail -2; echo LOCAL-HEAD:; git rev-parse HEAD; echo REMOTE-MAIN:; git rev-parse origin/main; echo BEHIND:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---LITE-V7-CORPUS---; ls data/lite_v7_corpus 2>/dev/null | wc -l; echo ---WORKDIR-DATA---; ls /home/arm/junyang/carbon/data/ 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1358C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358C; echo ---LITE-V7-INDEX-DS---; ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | head -25; ls /home/arm/junyang/carbon/data/lite_v7_index 2>/dev/null; ls /home/arm/junyang/carbon/data/ds_lite_v7 2>/dev/null; ls /home/arm/junyang/carbon-repo/data/ 2>/dev/null | grep -i lite; echo ---WEIGHTS-MTIME---; ls -lt /home/arm/junyang/carbon/data/weights 2>/dev/null | head -5; ls -lt /home/arm/junyang/carbon-repo/data/weights 2>/dev/null | head -5; echo ---TRAIN-LOGS---; ls -lt /home/arm/junyang/carbon/data/*train* /home/arm/junyang/carbon/*train* 2>/dev/null | head -10; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1358D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358D; python3 -c "
import json,glob,os
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\"))+sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/meta.json\"))+sorted(glob.glob(\"/home/arm/junyang/carbon-repo/data/*/index.json\")):
    try:
        d=json.load(open(p))
        n=len(d) if isinstance(d,(list,dict)) else d.get(\"n\",d.get(\"games\",d.get(\"samples\",\"?\")))
        print(p, n)
    except Exception as e:
        print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1358E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1358E; df -h / | tail -1; echo ---REPO-ROOT-NEW---; ls -lt --time-style=+%H:%M /home/arm/junyang/carbon-repo/ 2>/dev/null | head -12; echo ---TASKBOOKS---; ls -lt /home/arm/junyang/carbon-repo/*.md 2>/dev/null | head -8; find /home/arm/junyang/carbon-repo -maxdepth 2 -name "*task*" -newermt "2026-11-13 13:28" 2>/dev/null | head; echo ---END-E---' 2>&1
echo "rcE=$?"
echo "DGXSTATUS1358-END $(date '+%H:%M:%S %Z')"
