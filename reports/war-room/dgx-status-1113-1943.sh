#!/bin/bash
# DGX status probe trip 2026-11-13 19:43 - READ-ONLY. Marker DGXSTATUS1943A-K+W.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus since 19:28: snap_70 calibration GO (M2 +280 over gate, Echo2 +237) relayed to L20,
# GO flag landed 19:41; it73 pool swap ~20:04 est, then Terminal-U cut at next boundary;
# second-batch corpora (hakimi31/lite-v8/m3s3z) pulling; 2 single replays pending after batch 2.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1943A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder|snap_70|pack" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1943B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1943C $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943C; echo ---SNAP70-WEIGHT---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/snap_70.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/snap_70.npz 2>/dev/null; echo ---SNAP70-PACK---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/snap_70.py 2>/dev/null; echo ---M9XJ570-PACKS---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1943D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943D; echo ---DISK---; df -h / | tail -1; echo ---TMP-LISTS---; ls -l /tmp/*matches.json 2>/dev/null; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1943E $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943E; cd /home/arm/junyang/carbon/data 2>/dev/null; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1943W $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943W; cd /home/arm/junyang/carbon-repo/data 2>/dev/null; for d in k4v3_corpus hakimi3pro_corpus fable6_corpus hakimi31_corpus lite-v8_corpus m3s3z_corpus; do echo -n "$d: "; ls "$d" 2>/dev/null | wc -l; echo -n " files, "; du -sb "$d" 2>/dev/null | cut -f1; echo " bytes"; done; echo ---SINGLE-REPLAYS---; ls -l k4v3_corpus/*0f680a55* fable6_corpus/*255dc1d8* 2>&1; echo ---END-W---' 2>&1
echo "rcW=$?"

echo "DGXSTATUS1943K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1943K; cd /home/arm/junyang/carbon-repo && timeout 120 git fetch origin main 2>&1 | tail -2; echo ---BEHIND---; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---NEW-COMMITS---; git log --oneline HEAD..origin/main 2>/dev/null | head -15; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS1943-END $(date '+%H:%M:%S %Z')"
