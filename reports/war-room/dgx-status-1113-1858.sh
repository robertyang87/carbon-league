#!/bin/bash
# DGX status probe trip 2026-11-13 18:58 - READ-ONLY. Marker DGXSTATUS1858A-K.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus: it570 r1 should be done ~18:48-18:49 (was ep16/20 @18:43, g1 done 18:39:46);
# verify r1 completion (proc gone? ep19/20 weight + log mtime), it70 EVAL still unlanded per war room,
# HEAD lag was 97 (all ladder-list syncs, no code lag).
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1858A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1858B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1858C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858C; echo ---M9XJ570-WEIGHTS---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>/dev/null; echo ---TRAIN-LOGS-TAIL---; tail -5 /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log 2>&1; echo ---; tail -5 /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log 2>&1; echo ---LOG-MTIME---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log /home/arm/junyang/carbon/data/dq_logs/replica_chain_m9xj570.log 2>&1; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1858D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858D; echo ---DISK---; df -h / | tail -1; echo ---M9XJ570-CORPUS---; ls /home/arm/junyang/carbon/data/m9xj570_corpus/ 2>/dev/null | wc -l; ls /home/arm/junyang/carbon-repo/data/m9xj570_corpus/ 2>/dev/null | wc -l; echo ---DS-META---; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/ds_m9xj570/meta.json\"));print(\"ds_m9xj570 games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>&1 | head -2; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1858E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858E; cd /home/arm/junyang/carbon/data 2>/dev/null; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays-dir="; ls "$d/replays" 2>/dev/null | wc -l; echo -n "$d top-gz="; ls "$d"/*.json.gz 2>/dev/null | wc -l; done; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1858K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1858K; cd /home/arm/junyang/carbon-repo && timeout 120 git fetch origin main 2>&1 | tail -2; echo ---BEHIND---; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---NEW-COMMITS---; git log --oneline HEAD..origin/main 2>/dev/null | head -15; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS1858-END $(date '+%H:%M:%S %Z')"
