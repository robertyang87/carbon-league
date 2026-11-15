#!/bin/bash
# DGX status probe trip 2026-11-13 20:43 - READ-ONLY. Marker DGXSTATUS2043A-K+W.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus since 20:13: it73 pool swap DONE 20:02:44 (new PID 27737, pool 11, Terminal-U cut at it74
# boundary); six-family incremental pull DONE 48 games (m3s3z 45, hakimi31 65, k4v3 74, lite-v8 152,
# fable6 78, hakimi3pro 70) awaiting API warehousing; remote main was fd2d4f97, local HEAD 002b262b.
# Step K is ls-remote ONLY (no git fetch) per task body read-only rule.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS2043A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder|snap_70|pack" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS2043B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS2043C $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043C; echo ---SNAP70-WEIGHT---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/snap_70.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/snap_70.npz 2>/dev/null; echo ---SNAP70-PACK---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>/dev/null; echo ---M9XJ570-PACKS---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS2043D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043D; echo ---DISK---; df -h / | tail -1; echo ---TMP-LISTS---; ls -l /tmp/*matches.json 2>/dev/null; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS2043E $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043E; cd /home/arm/junyang/carbon/data 2>/dev/null; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS2043W $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043W; cd /home/arm/junyang/carbon-repo/data 2>/dev/null; for d in k4v3_corpus hakimi3pro_corpus fable6_corpus hakimi31_corpus lite-v8_corpus m3s3z_corpus; do echo -n "$d: "; ls "$d" 2>/dev/null | wc -l; echo -n " files, "; du -sb "$d" 2>/dev/null | cut -f1; echo " bytes"; done; echo ---END-W---' 2>&1
echo "rcW=$?"

echo "DGXSTATUS2043K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2043K; cd /home/arm/junyang/carbon-repo && echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS2043-END $(date '+%H:%M:%S %Z')"
