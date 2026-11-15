#!/bin/bash
# DGX status probe trip 2026-11-13 21:13 - READ-ONLY. Marker DGXSTATUS2113A-K+F+G.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Focus: it570 wr1 (PID 785811, est finish 21:07-21:10 last trip) - done or finishing?
# Also: producer-degeneracy dispatch to DGX line just landed in DGX chat (read-only note).
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS2113A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder|snap_70|pack|wr1" | grep -v grep | head -30; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS2113B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS2113C $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113C; echo ---SNAP70-WEIGHT---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/snap_70.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/snap_70.npz 2>/dev/null; echo ---M9XJ570-PACKS---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570r1.py /home/arm/junyang/carbon-repo/code/exp/m9xj570g1.py 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS2113D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113D; echo ---DISK---; df -h / | tail -1; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS2113E $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113E; cd /home/arm/junyang/carbon/data 2>/dev/null; for d in m3s2_corpus carago_corpus fable_corpus; do echo -n "$d replays="; ls "$d/replays" 2>/dev/null | wc -l; done; for f in m3s2_corpus/index.json carago_corpus/index.json fable_corpus/index.json; do echo -n "$f: "; python3 -c "import json;d=json.load(open(\"$f\"));print(len(d) if isinstance(d,list) else len(d.get(\"entries\",d)))" 2>/dev/null || echo MISSING; done; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS2113F $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113F; echo ---WR1-FILES---; find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -maxdepth 4 \( -iname "*wr1*" -o -iname "*m9xj570wr1*" \) -exec ls -l --time-style=long-iso {} \; 2>/dev/null | head -40; echo ---WR1-PACK---; stat -c "%n %s %y" /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py 2>&1; md5sum /home/arm/junyang/carbon-repo/code/exp/m9xj570wr1.py 2>/dev/null; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS2113G $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113G; echo ---WR1-LOG---; ls -lt /home/arm/junyang/carbon/data/weights/cnet2_m9xj570wr1* /home/arm/junyang/carbon/data/weights/*wr1* 2>/dev/null | head -25; echo ---LOGJSONL-TAIL---; for f in /home/arm/junyang/carbon/data/weights/*wr1*log* /home/arm/junyang/carbon/logs/*wr1* /home/arm/junyang/carbon/*wr1*.log; do [ -f "$f" ] && { echo "== $f"; tail -n 25 "$f"; }; done; echo ---ALL-EPOCH-NPZ---; ls -l --time-style=long-iso /home/arm/junyang/carbon/data/weights/ 2>/dev/null | grep -i wr1 | head -30; echo ---END-G---' 2>&1
echo "rcG=$?"

echo "DGXSTATUS2113K $(date '+%H:%M:%S %Z')"
timeout 180 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS2113K; cd /home/arm/junyang/carbon-repo && echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---END-K---' 2>&1
echo "rcK=$?"

echo "DGXSTATUS2113-END $(date '+%H:%M:%S %Z')"
