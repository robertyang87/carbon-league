#!/bin/bash
# DGX status report probe 2026-11-14 10:58 CST. Read-only. Marker DGXSTATUS11141058.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-1114-1058.log
ssh_dgx() {
  timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 "$1"
}
{
echo "DGXSTATUS11141058 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== A system/gpu/procs ==="
ssh_dgx 'echo DGXSTATUS11141058A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo "-- procs:"; ps -eo pid,etime,args | grep -E "[t]rain|[s]creen|[k]0_|spawn_main|[g]it (fetch|pull|merge|checkout)" | head -20; echo "-- spawn_main count:"; pgrep -f "spawn_mai[n]" | wc -l; echo "-- git procs:"; ps -eo pid,etime,args | grep -E "[g]it (fetch|pull|merge|checkout|stash)" || echo NO-GIT-PROC; echo "-- bigeval:"; ls -l /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1; md5sum /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1' 2>&1
echo "A SSH-RC=$?"
echo "=== B repo HEAD ==="
ssh_dgx 'echo DGXSTATUS11141058B; cd /home/arm/junyang/carbon-repo && git rev-parse HEAD; echo "-- stash:"; git stash list | wc -l; echo "-- status:"; git status --porcelain | head -5; echo "-- remote ls-remote:"; timeout 60 git ls-remote origin main | cut -f1; echo "-- fetch (bounded, refs only):"; timeout 60 git fetch origin main 2>&1 | tail -2; echo "-- behind count:"; git rev-list --count HEAD..origin/main 2>/dev/null || echo COUNT-FAILED' 2>&1
echo "B SSH-RC=$?"
echo "=== C corpora ==="
ssh_dgx 'echo DGXSTATUS11141058C; cd /home/arm/junyang/carbon; for d in data/m3s2_corpus data/carago_corpus data/fable_corpus; do echo "-- $d:"; ls $d/replays 2>/dev/null | wc -l; python3 -c "import json;print(len(json.load(open(\"$d/index.json\"))))" 2>/dev/null || echo "index.json MISSING"; done; for ds in ds_carago ds_fable; do echo "-- $ds:"; python3 -c "import json;print(json.load(open(\"data/$ds/meta.json\")))" 2>/dev/null || echo "meta.json MISSING"; done; echo "-- four-gap corpora:"; for c in m3s3z lite-v8 k4v3 krypton; do printf "%s replays=" "$c"; ls data/${c}_corpus/replays 2>/dev/null | wc -l; if [ -f data/${c}_corpus/index.json ]; then echo "  index PRESENT"; else echo "  index MISSING"; fi; done; echo "-- newest corpus mtimes:"; ls -lt data/*/replays 2>/dev/null | head -3; find data -name "*.json.gz" -newermt "2026-11-14 10:44" 2>/dev/null | head -5; echo "-- new files since 10:44 in home:"; find /home/arm/junyang -maxdepth 1 -newermt "2026-11-14 10:44" -type f 2>/dev/null | head -10' 2>&1
echo "C SSH-RC=$?"
echo "=== D k0 pickup state re-verify ==="
ssh_dgx 'echo DGXSTATUS11141058D; cd /home/arm/junyang/carbon-repo; echo "-- k0_train md5 x3:"; md5sum code/k0_train.py /home/arm/junyang/carbon/code/k0_train.py /home/arm/junyang/s26stage/code/k0_train.py 2>&1; echo "-- certify md5 x3:"; md5sum code/certify_pack.py /home/arm/junyang/carbon/code/certify_pack.py /home/arm/junyang/s26stage/code/certify_pack.py 2>&1; echo "-- screen logs:"; ls -l /home/arm/junyang/k0_screen_a0.log /home/arm/junyang/k0_screen_a1.log 2>&1; echo "-- backup dir count:"; ls /home/arm/junyang/ff_blockers_backup_20261114/ 2>/dev/null | wc -l; echo "-- k0 outs:"; ls -l /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out /home/arm/junyang/k0_b0_out 2>&1' 2>&1
echo "D SSH-RC=$?"
echo "DGXSTATUS11141058 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
