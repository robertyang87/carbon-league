#!/bin/bash
# DGX status probe trip 14:28 - read-only. Marker DGXSTATUS1428A-G.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50).
# Purpose: Lite-V7 dual-arm progress (g1 ep19 expected ~14:30); silent-corruption checks;
# locate k4v2r1 validation-field raw lines (seeds 993000-993019) for C2 1b case (read-only locate).
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1428A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1428B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428B; cd /home/arm/junyang/carbon-repo && git branch --show-current; timeout 60 git fetch origin 2>&1 | tail -2; echo LOCAL-HEAD:; git rev-parse HEAD; echo ORIGIN-MAIN:; git rev-parse origin/main; echo BEHIND-MAIN:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ORIGIN-V12:; git rev-parse origin/cnet2-search-v12 2>/dev/null; echo LS-REMOTE-V12:; timeout 30 git ls-remote origin refs/heads/cnet2-search-v12 2>&1; echo ---BUILD-TRAIN-MD5---; md5sum code/cnet2_build.py code/cnet2_train.py 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1428C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428C; echo ---LITE-V7-LOGJSONL---; cat /home/arm/junyang/carbon/data/weights/cnet2_lite_v7r1_log.jsonl; echo ---G1---; cat /home/arm/junyang/carbon/data/weights/cnet2_lite_v7g1_log.jsonl; echo ---REPLICA-CHAIN-TAIL---; tail -15 /home/arm/junyang/carbon/data/dq_logs/replica_chain_lite_v7.log; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1428D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428D; ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i lite | head -10; echo ---IT320-MD5---; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320g1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz 2>/dev/null; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1428E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428E; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\"))+sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/meta.json\")):
    try:
        d=json.load(open(p))
        n=len(d) if isinstance(d,(list,dict)) else d.get(\"n\",d.get(\"games\",d.get(\"samples\",\"?\")))
        print(p, n)
    except Exception as e:
        print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1428F $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428F; df -h / | tail -1; echo ---NEW-FILES-SINCE-1413---; find /home/arm/junyang/carbon-repo -maxdepth 2 -newermt "2026-11-13 14:13" 2>/dev/null | head -20; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS1428G $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1428G; echo ---K4V2R1-VALIDATION-RAW---; grep -rl "993000" /home/arm/junyang/carbon/data/ /home/arm/junyang/carbon-repo/data/ 2>/dev/null | head -10; echo ---SEED-FILES---; ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | grep -i -E "k4|valid|993" | head -10; find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 -iname "*k4v2*" 2>/dev/null | head -15; echo ---END-G---' 2>&1
echo "rcG=$?"

echo "DGXSTATUS1428-END $(date '+%H:%M:%S %Z')"
