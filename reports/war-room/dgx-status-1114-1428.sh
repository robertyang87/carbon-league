#!/bin/bash
# DGX status probe trip 2026-11-14 14:28 - read-only. Markers DGXSTATUS11141428A-E.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13).
# State at 14:14: Lite-V8 landed+warehoused (320d3ec7), GPU 0%/37C, packaging standby,
# corpus m3s2 152 / carago 183 / fable 313 / fable6 225 / krypton 111 / lite-v8 241,
# HEAD pinned 1d57e671, remote main 320d3ec7.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS11141428A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS11141428A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|screen|selfplay|build" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---SPAWN-BRACKET---; pgrep -f "spawn_mai[n]" | wc -l; echo ---ALL-PYTHON---; ps -eo pid,etime,args | grep -E "python3? " | grep -v grep | head -10; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS11141428B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS11141428B; cd /home/arm/junyang/carbon-repo && echo LOCAL-HEAD:; git rev-parse HEAD; echo REMOTE-MAIN-LSREMOTE:; timeout 60 git ls-remote origin main; echo ---STATUS---; git status --short | head -20; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS11141428C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS11141428C; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus fable6_corpus krypton-search256_corpus lite-v8_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---INDEX-JSON---; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---DS-META---; for d in ds_carago ds_fable; do echo "== $d"; python3 -c "
import json,os
p=\"/home/arm/junyang/carbon/data/$d/meta.json\"
d=json.load(open(p)); print(p, d.get(\"total\", d.get(\"games\", d.get(\"samples\"))), \"mtime\", __import__(\"datetime\").datetime.fromtimestamp(os.path.getmtime(p)).strftime(\"%m-%d %H:%M\"))
" 2>/dev/null || ls -lt /home/arm/junyang/carbon/data/$d 2>/dev/null | head -3; done; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS11141428D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS11141428D; echo ---FOUR-PACKS-MD5---; cd /home/arm/junyang/carbon-repo && ls code/exp/ | grep -iE "a4|base" | head; md5sum code/exp/*a4* code/exp/*base* 2>/dev/null | head -8; echo ---LITEV8-MANIFEST---; wc -l /home/arm/junyang/litev8/pull_lite-v8.manifest 2>/dev/null; ls -l --time-style=+%m-%d_%H:%M /home/arm/junyang/litev8/pull_lite-v8.manifest 2>/dev/null; echo ---GAP5-MANIFEST---; wc -l /home/arm/junyang/gap5/pull0033_fable_gap5.manifest 2>/dev/null; echo ---B0-LOG---; find /home/arm/junyang/carbon -maxdepth 2 -name "*k0_screen_b0*" 2>/dev/null | head -3; ls -l --time-style=+%m-%d_%H:%M /home/arm/junyang/carbon/data/k0_screen_b0.log 2>/dev/null; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS11141428E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS11141428E; df -h / | tail -1; echo ---NEWEST-DATA---; ls -lt --time-style=+%m-%d_%H:%M /home/arm/junyang/carbon/data/ 2>/dev/null | head -8; echo ---NEWEST-HOME---; ls -lt --time-style=+%m-%d_%H:%M /home/arm/junyang/ 2>/dev/null | head -8; echo ---NEW-FILES-SINCE-1414---; find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon/data -maxdepth 2 -newermt "2026-11-14 14:14" 2>/dev/null | head -15; echo ---END-E---' 2>&1
echo "rcE=$?"
echo "DGXSTATUS11141428-END $(date '+%H:%M:%S %Z')"
