#!/bin/bash
# DGX status probe trip 2026-11-13 19:13 - READ-ONLY. Marker DGXSTATUS1913A-K.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus: it570 packs built + in main 9ef9c031, calibration orders r1->C1 / g1->C2 dispatched;
# verify pack files on DGX (md5), GPU idle, no new training, HEAD lag was 98 (docs only), it70 EVAL is L20-side.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1913A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1913B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1913C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913C; echo ---M9XJ570-WEIGHTS---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>/dev/null; echo ---PACK-FILES---; stat -c "%n %s %y" /home/arm/junyang/carbon/code/exp/m9xj570r1.py /home/arm/junyang/carbon/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon/code/exp/m9xj570r1.py /home/arm/junyang/carbon/code/exp/m9xj570g1.py 2>/dev/null; echo ---TRAIN-LOGS-TAIL---; tail -3 /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log 2>&1; echo ---; tail -3 /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log 2>&1; echo ---LOG-MTIME---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log 2>&1; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1913D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913D; echo ---DISK---; df -h / | tail -1; echo ---DS-META---; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/ds_m9xj570/meta.json\"));print(\"ds_m9xj570 games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>&1 | head -2; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1913E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913E; cd /home/arm/junyang/carbon/data 2>/dev/null; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1913K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1913K; cd /home/arm/junyang/carbon-repo && timeout 120 git fetch origin main 2>&1 | tail -2; echo ---BEHIND---; git rev-list --count HEAD..origin/main 2>/dev/null; echo ---NEW-COMMITS---; git log --oneline HEAD..origin/main 2>/dev/null | head -15; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS1913-END $(date '+%H:%M:%S %Z')"
