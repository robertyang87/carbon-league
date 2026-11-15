#!/bin/bash
# DGX status probe trip 2026-11-13 19:28 - READ-ONLY. Marker DGXSTATUS1928A-K.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus: snap_70 weight landed on DGX 19:28 (md5 22c5075b..., two-hop verified per DGX-line exec);
# verify on DGX + pack build status; it570 calibration both FAIL (r1 -407, g1 +95.9) per C1/C2;
# it70 EVAL landed 19:24:39 (+948.4/+524.6); new 3 corpora k4v3/hakimi3pro/fable6 pulled.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1928A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder|snap_70|pack" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1928B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1928C $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928C; echo ---SNAP70-WEIGHT---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/snap_70.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/snap_70.npz 2>/dev/null; echo ---WEIGHTS-RECENT---; ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/ 2>/dev/null | head -8; echo ---SNAP70-PACK---; ls -lt --time-style=+%F_%T /home/arm/junyang/carbon-repo/code/exp/ 2>/dev/null | head -10; echo ---M9XJ570-PACKS---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1928D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928D; echo ---DISK---; df -h / | tail -1; echo ---DS-M9XJ570---; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/ds_m9xj570/meta.json\"));print(\"ds_m9xj570 games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>&1 | head -2; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1928E $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928E; cd /home/arm/junyang/carbon/data 2>/dev/null; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json k4v3_corpus/index.json hakimi3pro_corpus/index.json fable6_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for d in m3s2_corpus carago_corpus fable_corpus k4v3_corpus hakimi3pro_corpus fable6_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1928K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1928K; cd /home/arm/junyang/carbon-repo && timeout 120 git fetch origin main 2>&1 | tail -2; echo ---BEHIND---; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---NEW-COMMITS---; git log --oneline HEAD..origin/main 2>/dev/null | head -15; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS1928-END $(date '+%H:%M:%S %Z')"
