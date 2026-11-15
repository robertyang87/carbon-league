#!/bin/bash
# DGX status probe 2026-11-14 13:03 CST. Read-only. Marker DGXSTATUS11141303.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-1303-ts.log
ssh_dgx() {
  timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 "$1"
}
{
echo "DGXSTATUS11141303 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== A system/gpu/procs ==="
ssh_dgx 'echo DGXSTATUS11141303A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo "-- procs:"; ps -eo pid,etime,args | grep -E "[a]4_select|[t]rain|[s]creen|[k]0_|spawn_main|[g]it (fetch|pull|merge|checkout)" | head -20; echo "-- spawn_main count:"; pgrep -f "spawn_mai[n]" | wc -l; echo "-- git procs:"; ps -eo pid,etime,args | grep -E "[g]it (fetch|pull|merge|checkout|stash)" || echo NO-GIT-PROC; echo "-- top cpu:"; ps -eo pid,etime,%cpu,args --sort=-%cpu | head -8' 2>&1
echo "A SSH-RC=$?"
echo "=== B repo HEAD (ls-remote only, no fetch) ==="
ssh_dgx 'echo DGXSTATUS11141303B; cd /home/arm/junyang/carbon-repo && git rev-parse HEAD; echo "-- status:"; git status --porcelain | head -5; echo "-- remote HEAD live:"; timeout 60 git ls-remote origin HEAD | cut -f1; echo "-- remote main live:"; timeout 60 git ls-remote origin main | cut -f1' 2>&1
echo "B SSH-RC=$?"
echo "=== C corpora ==="
ssh_dgx 'echo DGXSTATUS11141303C; cd /home/arm/junyang/carbon; for d in data/m3s2_corpus data/carago_corpus data/fable_corpus data/fable6_corpus; do echo "-- $d:"; ls $d/replays 2>/dev/null | wc -l; python3 -c "import json;print(len(json.load(open(\"$d/index.json\"))))" 2>/dev/null || echo "index.json MISSING"; done; for ds in ds_carago ds_fable; do echo "-- $ds total:"; python3 -c "import json;print(json.load(open(\"data/$ds/meta.json\"))[\"total\"])" 2>/dev/null || echo "meta.json MISSING"; done' 2>&1
echo "C SSH-RC=$?"
echo "=== D four packs ==="
ssh_dgx 'echo DGXSTATUS11141303D; md5sum /home/arm/junyang/carbon-repo/code/exp/m3s2g1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2g1_base.py /home/arm/junyang/carbon-repo/code/exp/m3s2r1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2r1_base.py 2>&1; ls -l /home/arm/junyang/carbon-repo/code/exp/m3s2g1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2g1_base.py /home/arm/junyang/carbon-repo/code/exp/m3s2r1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2r1_base.py 2>&1; echo "-- new exp files since 12:58:"; find /home/arm/junyang/carbon-repo/code/exp -newermt "2026-11-14 12:58" -type f 2>/dev/null; echo "-- new files home maxdepth2 since 12:58:"; find /home/arm/junyang -maxdepth 2 -newermt "2026-11-14 12:58" -type f 2>/dev/null | head -15' 2>&1
echo "D SSH-RC=$?"
echo "DGXSTATUS11141303 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
