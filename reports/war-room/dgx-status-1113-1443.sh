#!/bin/bash
# DGX status probe trip 2026-11-13 14:43 - read-only. Marker DGXSTATUS1443A-E.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50).
# Purpose: Lite-V7 r1 ep19 landing (est ~14:40) + weights md5/val; Hakimi v3 handover
# precondition (frozen 105 games); silent-corruption checks.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1443A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1443B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443B; cd /home/arm/junyang/carbon-repo && git branch --show-current; timeout 60 git fetch origin 2>&1 | tail -2; echo LOCAL-HEAD:; git rev-parse HEAD; echo ORIGIN-MAIN:; git rev-parse origin/main; echo BEHIND-MAIN:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---BUILD-TRAIN-MD5---; md5sum code/cnet2_build.py code/cnet2_train.py 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1443C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443C; echo ---R1-LOG---; tail -8 /home/arm/junyang/carbon/data/weights/cnet2_lite_v7r1_log.jsonl; echo ---G1-LOG-TAIL---; tail -3 /home/arm/junyang/carbon/data/weights/cnet2_lite_v7g1_log.jsonl; echo ---LITE-WEIGHTS---; ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i lite | head -12; echo ---R1-EP19-MD5---; md5sum /home/arm/junyang/carbon/data/weights/cnet2_lite_v7r1.npz 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1443D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443D; echo ---HAKIMI3---; ls /home/arm/junyang/carbon/data/hakimi3_corpus 2>/dev/null | wc -l; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1443E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1443E; df -h / | tail -1; echo ---NEW-FILES-SINCE-1428---; find /home/arm/junyang/carbon-repo -maxdepth 2 -newermt "2026-11-13 14:28" 2>/dev/null | head -20; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1443-END $(date '+%H:%M:%S %Z')"
