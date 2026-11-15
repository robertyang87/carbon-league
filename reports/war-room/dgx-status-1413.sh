#!/bin/bash
# DGX status probe trip 14:13 - read-only. Marker DGXSTATUS1413A-G.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50).
# Purpose: Lite-V7 dual-arm training progress since 14:03 launch; silent-corruption checks.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1413A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1413B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413B; cd /home/arm/junyang/carbon-repo && timeout 60 git fetch origin 2>&1 | tail -2; echo LOCAL-HEAD:; git rev-parse HEAD; echo REMOTE-MAIN:; git rev-parse origin/main; echo BEHIND:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---LITE-V7-REPO-CORPUS---; ls data/lite_v7_corpus 2>/dev/null | wc -l; echo ---HAKIMI-REPO-CORPUS---; ls data/hakimi3_corpus 2>/dev/null | wc -l; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1413C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413C; echo ---LITE-V7-LOGS---; ls -lt /home/arm/junyang/carbon/data/*lite_v7* /home/arm/junyang/carbon/*lite_v7* 2>/dev/null | head -15; echo ---LITE-V7-WEIGHTS---; ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i lite | head -8; echo ---REPLICA-CHAIN-LOG-TAIL---; tail -25 /home/arm/junyang/carbon/data/replica_chain_lite_v7.log 2>/dev/null || find /home/arm/junyang/carbon -maxdepth 3 -name "*replica_chain*lite*" 2>/dev/null | head; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1413D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413D; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\"))+sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/meta.json\")):
    try:
        d=json.load(open(p))
        n=len(d) if isinstance(d,(list,dict)) else d.get(\"n\",d.get(\"games\",d.get(\"samples\",\"?\")))
        print(p, n)
    except Exception as e:
        print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---DS-META---; for d in ds_lite_v7 ds_carago ds_fable; do echo "== $d"; ls -lt /home/arm/junyang/carbon/data/$d 2>/dev/null | head -4; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1413E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413E; df -h / | tail -1; echo ---REPO-ROOT-NEW---; ls -lt --time-style=+%H:%M /home/arm/junyang/carbon-repo/ 2>/dev/null | head -12; echo ---NEW-FILES-SINCE-1358---; find /home/arm/junyang/carbon-repo -maxdepth 2 -newermt "2026-11-13 13:58" 2>/dev/null | head -20; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1413F $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413F; echo ---R1-LOG-TAIL---; for f in /home/arm/junyang/carbon/data/dq_logs/*lite_v7r1* /home/arm/junyang/carbon/data/*lite_v7r1*.log /home/arm/junyang/carbon/data/logs/*lite_v7r1* ; do echo "== $f"; tail -6 "$f" 2>/dev/null; done; ls -lt /home/arm/junyang/carbon/data/ 2>/dev/null | grep -i -E "lite|r1|g1" | head -10; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS1413G $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1413G; echo ---G1-LOG-TAIL---; for f in /home/arm/junyang/carbon/data/dq_logs/*lite_v7g1* /home/arm/junyang/carbon/data/*lite_v7g1*.log /home/arm/junyang/carbon/data/logs/*lite_v7g1*; do echo "== $f"; tail -6 "$f" 2>/dev/null; done; echo ---TRAIN-JSONL---; ls -lt /home/arm/junyang/carbon/data/*train*.jsonl /home/arm/junyang/carbon/data/*lite*.jsonl 2>/dev/null | head -5; echo ---END-G---' 2>&1
echo "rcG=$?"
echo "DGXSTATUS1413-END $(date '+%H:%M:%S %Z')"
