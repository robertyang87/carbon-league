#!/bin/bash
# DGX status report probe 2026-11-14 11:43 CST. Read-only. Marker DGXSTATUS11141143.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-1114-1143.log
ssh_dgx() {
  timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 "$1"
}
{
echo "DGXSTATUS11141143 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== A system/gpu/procs ==="
ssh_dgx 'echo DGXSTATUS11141143A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo "-- procs:"; ps -eo pid,etime,args | grep -E "[a]4_select|[t]rain|[s]creen|[k]0_|spawn_main|[g]it (fetch|pull|merge|checkout)" | head -20; echo "-- spawn_main count:"; pgrep -f "spawn_mai[n]" | wc -l; echo "-- git procs:"; ps -eo pid,etime,args | grep -E "[g]it (fetch|pull|merge|checkout|stash)" || echo NO-GIT-PROC; echo "-- bigeval:"; ls -l /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1; md5sum /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1' 2>&1
echo "A SSH-RC=$?"
echo "=== B repo HEAD (ls-remote only, no fetch) ==="
ssh_dgx 'echo DGXSTATUS11141143B; cd /home/arm/junyang/carbon-repo && git rev-parse HEAD; echo "-- stash:"; git stash list | wc -l; echo "-- status:"; git status --porcelain | head -5; echo "-- remote ls-remote:"; timeout 60 git ls-remote origin main | cut -f1; echo "-- behind count vs origin/main ref:"; git rev-list --count HEAD..origin/main 2>/dev/null || echo COUNT-FAILED' 2>&1
echo "B SSH-RC=$?"
echo "=== C corpora ==="
ssh_dgx 'echo DGXSTATUS11141143C; cd /home/arm/junyang/carbon; for d in data/m3s2_corpus data/carago_corpus data/fable_corpus; do echo "-- $d:"; ls $d/replays 2>/dev/null | wc -l; python3 -c "import json;print(len(json.load(open(\"$d/index.json\"))))" 2>/dev/null || echo "index.json MISSING"; done; for ds in ds_carago ds_fable; do echo "-- $ds total:"; python3 -c "import json;print(json.load(open(\"data/$ds/meta.json\"))[\"total\"])" 2>/dev/null || echo "meta.json MISSING"; done; echo "-- new files since 11:28 in home:"; find /home/arm/junyang -maxdepth 2 -newermt "2026-11-14 11:28" -type f 2>/dev/null | head -15' 2>&1
echo "C SSH-RC=$?"
echo "=== D A4 packing-standby re-verify ==="
ssh_dgx 'echo DGXSTATUS11141143D; echo "-- a4 dir:"; ls -la /home/arm/junyang/a4_selection_20261114/ 2>&1; echo "-- VOID marker:"; cat /home/arm/junyang/a4_selection_20261114/VOID_20261114_1136.txt 2>&1; echo "-- r1_sel:"; ls /home/arm/junyang/a4_selection_20261114/r1_sel/ 2>&1; echo "-- any selection json:"; ls /home/arm/junyang/a4_selection_20261114/*selection*.json 2>&1; echo "-- pack staging:"; md5sum /home/arm/junyang/a4_pack_stage/cnet2_pack.py /home/arm/junyang/a4_pack_stage/cnet2_search_pack.py 2>&1; echo "-- a4 log tail:"; tail -3 /home/arm/junyang/a4_select_20261114.log 2>&1; ls -l /home/arm/junyang/a4_select_20261114.log' 2>&1
echo "D SSH-RC=$?"
echo "DGXSTATUS11141143 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
