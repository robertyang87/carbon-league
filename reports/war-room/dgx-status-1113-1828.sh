#!/bin/bash
# DGX status probe trip 2026-11-13 18:28 - READ-ONLY. Marker DGXSTATUS1828A-K.
# Channel: Tailscale SSH arm@100.84.148.73 (cloud-assistant retired 2026-11-13 ~10:50, owner order).
# Since 18:13: it570 corpus pull finished 18:20 (89/89, diff 0); gate v1.5 brushed on DGX clone
# (13d89867, ff to 002b262b); Stage B pool rebuilt (data/stageB_pool.json d6916c80, 1,525 games);
# USAGE_MARK.json landed in two places (48cd3644...); ds_m9xj570 built + dual-arm training launched
# 18:23:59 (PID 713670 r1 / 713671 g1, GPU 96%, r1 ep1 / g1 ep3 @18:25 per DGX line).
# Verify: training three-piece (ps + GPU + weight/log mtime), epoch progress, gate/pool/mark files.
KEY="$HOME/.ssh/id_ed25519"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p")
DGX="arm@100.84.148.73"

echo "DGXSTATUS1828A $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader; echo ---TRAIN-PS---; ps -eo pid,etime,args | grep -E "train|cnet2|policy_verify|bigeval|gate|build|index|selfplay|m9xj570|ladder" | grep -v grep | head -20; echo ---GIT-PS---; ps -eo pid,etime,args | grep -E "git (fetch|pull|checkout|clone)" | grep -v grep; echo ---END-A---' 2>&1
echo "rcA=$?"

echo "DGXSTATUS1828B $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828B; cd /home/arm/junyang/carbon-repo && git branch --show-current; echo LS-REMOTE-MAIN:; timeout 60 git ls-remote origin main 2>&1 | tail -1; echo LOCAL-HEAD:; git rev-parse HEAD; echo ---HEAD-LOG---; git log --oneline -3 2>/dev/null; echo ---BEHIND-COUNT---; git rev-list --count HEAD..$(timeout 60 git ls-remote origin main 2>/dev/null | awk "{print \$1}") 2>/dev/null; echo ---BIGEVAL-STATUS---; git status --short data/bigeval.jsonl 2>/dev/null; echo ---END-B---' 2>&1
echo "rcB=$?"

echo "DGXSTATUS1828C $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828C; echo ---M9XJ570-WEIGHTS---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj570r1.npz /home/arm/junyang/carbon/data/weights/cnet2_m9xj570g1.npz 2>/dev/null; echo ---TRAIN-LOGS-TAIL---; tail -3 /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log 2>&1; echo ---; tail -3 /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log 2>&1; echo ---LOG-MTIME---; stat -c "%n %s %y" /home/arm/junyang/carbon/data/dq_logs/m9xj570r1.log /home/arm/junyang/carbon/data/dq_logs/m9xj570g1.log /home/arm/junyang/carbon/data/dq_logs/replica_chain_m9xj570.log 2>&1; echo ---END-C---' 2>&1
echo "rcC=$?"

echo "DGXSTATUS1828D $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828D; python3 -c "
import json,glob
for p in sorted(glob.glob(\"/home/arm/junyang/carbon/data/*/index.json\")):
    try:
        d=json.load(open(p)); print(p, len(d) if isinstance(d,(list,dict)) else d)
    except Exception as e: print(p, \"ERR\", e)
"; echo ---CORPUS-COUNTS---; for c in m3s2_corpus carago_corpus fable_corpus lite_v7_corpus hakimi3_corpus m9xj570_corpus; do echo -n "$c: "; ls /home/arm/junyang/carbon/data/$c 2>/dev/null | wc -l; done; echo ---DS-M9XJ570-META---; python3 -c "import json; m=json.load(open(\"/home/arm/junyang/carbon/data/ds_m9xj570/meta.json\")); print(m)" 2>&1 | head -c 500; echo; echo ---END-D---' 2>&1
echo "rcD=$?"

echo "DGXSTATUS1828E $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828E; df -h / | tail -1; echo ---END-E---' 2>&1
echo "rcE=$?"

echo "DGXSTATUS1828F $(date '+%H:%M:%S %Z')"
timeout 120 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828F; cd /home/arm/junyang/carbon-repo && echo ---S1-MD5-REPO---; md5sum code/stage0_teacher.py code/cnet2_search.py code/packs/cnet2_search.py code/stage1_teacher_gate.py code/stage0_guided_match.py code/cnet2_np.py code/packs/cnet2_np.py 2>/dev/null; echo ---P1-MD5-REPO---; md5sum code/cnet2_build.py code/cnet2_train.py code/set_uw.py 2>/dev/null; echo ---GATE-STAT---; stat -c "%n %s %y" code/stage1_teacher_gate.py 2>/dev/null; echo ---END-F---' 2>&1
echo "rcF=$?"

echo "DGXSTATUS1828G $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828G; echo ---STAGEB-POOL-NEW---; ls -l /home/arm/junyang/carbon/data/stageB_pool.json 2>&1; md5sum /home/arm/junyang/carbon/data/stageB_pool.json 2>/dev/null; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon/data/stageB_pool.json\")); print(\"keys:\", sorted(d.keys())[:12]); print(\"total:\", d.get(\"total\") or d.get(\"pool_size\") or d.get(\"count\"))" 2>&1; echo ---STAGEB-POOL-OLD---; ls -l /home/arm/junyang/carbon/data/stageB_candidate_pool.json 2>&1; echo ---USAGE-MARK---; md5sum /home/arm/junyang/carbon/data/m9xj570_corpus/USAGE_MARK.json /home/arm/junyang/carbon-repo/data/m9xj570_corpus/USAGE_MARK.json 2>&1; stat -c "%n %s %y" /home/arm/junyang/carbon/data/m9xj570_corpus/USAGE_MARK.json /home/arm/junyang/carbon-repo/data/m9xj570_corpus/USAGE_MARK.json 2>&1; echo ---END-G---' 2>&1
echo "rcG=$?"

echo "DGXSTATUS1828H $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828H; echo ---M9XJ320R1-NPZ---; ls -l /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz 2>&1; md5sum /home/arm/junyang/carbon/data/weights/cnet2_m9xj320r1.npz 2>/dev/null; echo ---END-H---' 2>&1
echo "rcH=$?"

echo "DGXSTATUS1828J $(date '+%H:%M:%S %Z')"
timeout 90 ssh "${SSH_OPTS[@]}" $DGX 'echo MARK-DGXSTATUS1828J; echo ---IT570-CORPUS-REPO---; ls /home/arm/junyang/carbon-repo/data/m9xj570_corpus/ 2>/dev/null | wc -l; echo ---IT570-MATCHES-FILE---; ls -l /home/arm/junyang/carbon-repo/data/m9xj570_matches.json 2>&1; python3 -c "import json; d=json.load(open(\"/home/arm/junyang/carbon-repo/data/m9xj570_matches.json\")); print(\"m9xj570_matches entries:\", len(d))" 2>&1; echo ---END-J---' 2>&1
echo "rcJ=$?"

echo "DGXSTATUS1828-END $(date '+%H:%M:%S %Z')"
