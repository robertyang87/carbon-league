#!/bin/bash
# DGX status report probe 2026-11-14 11:58 CST. Read-only. Marker DGXSTATUS11141158.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-1114-1158.log
ssh_dgx() {
  timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 "$1"
}
{
echo "DGXSTATUS11141158 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== A system/gpu/procs ==="
ssh_dgx 'echo DGXSTATUS11141158A; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo "-- procs:"; ps -eo pid,etime,args | grep -E "[a]4_select|[t]rain|[s]creen|[k]0_|spawn_main|[g]it (fetch|pull|merge|checkout)" | head -20; echo "-- spawn_main count:"; pgrep -f "spawn_mai[n]" | wc -l; echo "-- git procs:"; ps -eo pid,etime,args | grep -E "[g]it (fetch|pull|merge|checkout|stash)" || echo NO-GIT-PROC; echo "-- bigeval:"; ls -l /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1; md5sum /home/arm/junyang/carbon-repo/data/bigeval.jsonl 2>&1' 2>&1
echo "A SSH-RC=$?"
echo "=== B repo HEAD (ls-remote only, no fetch) ==="
ssh_dgx 'echo DGXSTATUS11141158B; cd /home/arm/junyang/carbon-repo && git rev-parse HEAD; echo "-- status:"; git status --porcelain | head -5; echo "-- remote HEAD live:"; timeout 60 git ls-remote origin HEAD | cut -f1; echo "-- remote main live:"; timeout 60 git ls-remote origin main | cut -f1; echo "-- local origin/main ref:"; git rev-parse origin/main 2>&1; echo "-- behind vs live remote main (ls-remote sha):"; L=$(timeout 60 git ls-remote origin main | cut -f1); git rev-list --count HEAD..$L 2>/dev/null || echo COUNT-FAILED' 2>&1
echo "B SSH-RC=$?"
echo "=== C corpora ==="
ssh_dgx 'echo DGXSTATUS11141158C; cd /home/arm/junyang/carbon; for d in data/m3s2_corpus data/carago_corpus data/fable_corpus; do echo "-- $d:"; ls $d/replays 2>/dev/null | wc -l; python3 -c "import json;print(len(json.load(open(\"$d/index.json\"))))" 2>/dev/null || echo "index.json MISSING"; done; for ds in ds_carago ds_fable; do echo "-- $ds total:"; python3 -c "import json;print(json.load(open(\"data/$ds/meta.json\"))[\"total\"])" 2>/dev/null || echo "meta.json MISSING"; done' 2>&1
echo "C SSH-RC=$?"
echo "=== D delivered items re-verify (g1 packs + certify V09) ==="
ssh_dgx 'echo DGXSTATUS11141158D; echo "-- exp packs:"; md5sum /home/arm/junyang/carbon-repo/code/exp/m3s2g1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2g1_base.py 2>&1; ls -l /home/arm/junyang/carbon-repo/code/exp/m3s2g1_a4.py /home/arm/junyang/carbon-repo/code/exp/m3s2g1_base.py 2>&1; echo "-- certify 3 places:"; md5sum /home/arm/junyang/carbon-repo/code/certify_pack.py /home/arm/junyang/carbon/code/certify_pack.py /home/arm/junyang/carbon/s26stage/code/certify_pack.py 2>&1; echo "-- pack staging:"; md5sum /home/arm/junyang/a4_pack_stage/cnet2_pack.py /home/arm/junyang/a4_pack_stage/cnet2_search_pack.py 2>&1; echo "-- verify jsons:"; ls -l /home/arm/junyang/m3s2g1_a4_verify.json /home/arm/junyang/m3s2g1_base_verify.json 2>&1' 2>&1
echo "D SSH-RC=$?"
echo "=== E A4 dir / r1 winner watch ==="
ssh_dgx 'echo DGXSTATUS11141158E; ls -la /home/arm/junyang/a4_selection_20261114/ 2>&1; echo "-- any selection json:"; ls /home/arm/junyang/a4_selection_20261114/*selection*.json 2>&1; echo "-- new files since 11:43 in home (maxdepth 2):"; find /home/arm/junyang -maxdepth 2 -newermt "2026-11-14 11:43" -type f 2>/dev/null | head -15; echo "-- a4 log tail:"; tail -3 /home/arm/junyang/a4_select_20261114.log 2>&1' 2>&1
echo "E SSH-RC=$?"
echo "DGXSTATUS11141158 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
