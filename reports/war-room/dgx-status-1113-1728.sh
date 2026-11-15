#!/bin/bash
# DGX status probe trip 2026-11-13 17:28 - READ-ONLY. Marker DGXSTATUS1728A-H.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Purpose: routine 15-min patrol. DGX expected idle standby (war-room ruling);
# remote main moved to 8fc10d07 (17:13 lag 21) - verify local HEAD
# still 7b025ae5 and v1.1 files intact; pull decision pending owner pilot ruling.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1728A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1728B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---HEAD-LOG---; git log --oneline -3 2>/dev/null; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1728C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728C; echo ---HAKIMI-WEIGHTS---; md5sum /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_hakimi3r1.npz /home/arm/junyang/carbon/data/weights/cnet2_hakimi3g1.npz 2>/dev/null; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1728D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728D; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---DS-META---; for d in ds_carago ds_fable ds_hakimi3; do echo -n "$d meta: "; python3 -c "import json,sys; m=json.load(open(\"/home/arm/junyang/carbon/data/$d/meta.json\")); print({k:m.get(k) for k in (\"n\",\"samples\",\"num_samples\",\"games\",\"num_games\") if k in m} or m)" 2>&1 | head -c 300; echo; done; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1728E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728E; df -h / | tail -1; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1728F $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728F; cd /home/arm/junyang/carbon-repo && echo ---S1-V11-MD5-REPO---; md5sum code/stage0_teacher.py code/cnet2_search.py code/packs/cnet2_search.py code/stage1_teacher_gate.py code/stage0_guided_match.py code/cnet2_np.py code/packs/cnet2_np.py 2>/dev/null; echo ---P1-MD5-REPO---; md5sum code/cnet2_build.py code/cnet2_train.py code/set_uw.py 2>/dev/null; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS1728G $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728G; ls -l /home/arm/junyang/carbon/data/stageB_candidate_pool.json 2>&1; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon/data/stageB_candidate_pool.json\")); print(\"stageB_pool_entries:\", len(d) if isinstance(d,list) else d)" 2>&1; echo ---END-G---' 2>&1
echo "rcG=$?"

echo "DGXSTATUS1728H $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1728H; echo ---M9XJ320R1-NPZ---; ls -l /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz 2>/dev/null; echo ---END-H---' 2>&1
echo "rcH=$?"

echo "DGXSTATUS1728-END $(date '+%H:%M:%S %Z')"
