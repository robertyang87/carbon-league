#!/bin/bash
# DGX status probe trip 2026-11-13 20:58 - READ-ONLY. Marker DGXSTATUS2058A-K+F+W.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus: war-room 20:53 dispatched it570 weighted pilot (wr1, sole executor DGX):
#   build ds_m9xj570_w from ds_m9xj570 (89-game frozen, 26,611 samples) -> train
#   cnet2_m9xj570wr1 attention single arm 20 epochs -> pack v2. Expect build/training
#   in progress ~5 min after dispatch. Also standard: GPU, ps, HEAD vs remote (ls-remote only).
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS2058A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder|snap_70|pack|wr1" | grep -v grep | head -30; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS2058B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS2058C $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058C; echo ---SNAP70-WEIGHT---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/snap_70.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/snap_70.npz 2>/dev/null; echo ---M9XJ570-PACKS---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS2058D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058D; echo ---DISK---; df -h / | tail -1; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS2058E $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058E; cd /home/arm/junyang/carbon/data 2>/dev/null; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; for ds in ds_carago ds_fable; do echo -n "$ds: "; python3 -c "import json;d=json.load(open(\"/home/arm/junyang/carbon/data/$ds/meta.json\"));print(\"games=\",d.get(\"n_games\",len(d.get(\"games\",[]))),\"samples=\",d.get(\"n_samples\",d.get(\"samples\")))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS2058F $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058F; echo ---DS-M9XJ570---; ls -lad /home/arm/junyang/carbon/data/ds_m9xj570* /home/arm/junyang/carbon-repo/data/ds_m9xj570* 2>&1; for d in /home/arm/junyang/carbon/data/ds_m9xj570 /home/arm/junyang/carbon/data/ds_m9xj570_w; do echo "== $d"; ls -l "$d" 2>/dev/null | head -20; python3 -c "import json,glob;f=glob.glob(\"$d/meta.json\");print(open(f[0]).read()[:600] if f else \"no meta.json\")" 2>/dev/null; done; echo ---WR1-FILES---; find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 \( -iname "*wr1*" -o -iname "*m9xj570wr1*" \) -exec ls -l --time-style=long-iso {} \; 2>/dev/null | head -30; echo ---WR1-LOG-TAIL---; ls -lt /home/arm/junyang/carbon/*.log /home/arm/junyang/carbon/logs/* 2>/dev/null | head -10; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS2058W $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058W; cd /home/arm/junyang/carbon-repo/data 2>/dev/null; for d in k4v3_corpus hakimi3pro_corpus fable6_corpus hakimi31_corpus lite-v8_corpus m3s3z_corpus; do echo -n "$d: "; ls "$d" 2>/dev/null | wc -l; echo -n " files, "; du -sb "$d" 2>/dev/null | cut -f1; echo " bytes"; done; echo ---END-W---' 2>&1
echo "rcW=$?"

echo "DGXSTATUS2058K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2058K; cd /home/arm/junyang/carbon-repo && echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS2058-END $(date '+%H:%M:%S %Z')"
