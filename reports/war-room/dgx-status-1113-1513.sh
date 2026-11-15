#!/bin/bash
# DGX status probe trip 2026-11-13 15:13 - read-only. Marker DGXSTATUS1513A-E.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50).
# Purpose: Hakimi v3 r1 epoch progress (g1 reportedly finished ~15:12, best ep13 0.8872)
# + ETA recompute; verify repo HEAD a5c6364 vs origin/main + third-round P1 md5
# (train 4140d398 / set_uw 0d0fbf10, build 030b317b); silent-corruption checks.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1513A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1513A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1513B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1513B; cd /home/arm/junyang/carbon-repo && git branch --show-current; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo BEHIND-MAIN:; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---P1-THIRD-MD5-REPO---; md5sum code/cnet2_build.py code/cnet2_train.py code/set_uw.py 2>/dev/null; echo ---P1-THIRD-MD5-WORKDIR---; md5sum /home/arm/junyang/carbon/code/cnet2_build.py /home/arm/junyang/carbon/code/cnet2_train.py /home/arm/junyang/carbon/code/set_uw.py 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1513C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1513C; echo ---R1-LOG---; tail -8 /home/arm/junyang/carbon/data/dq_logs/hakimi3r1.log; echo ---G1-LOG---; tail -8 /home/arm/junyang/carbon/data/dq_logs/hakimi3g1.log; echo ---HAKIMI-WEIGHTS---; ls -lt /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i hakimi3 | head -20; echo ---HAKIMI-BEST-MD5---; md5sum /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1513D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1513D; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1513E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1513E; df -h / | tail -1; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1513-END $(date '+%H:%M:%S %Z')"
