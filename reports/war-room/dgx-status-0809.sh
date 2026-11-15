#!/bin/bash
# DGX read-only status probe 08:09 (war-room-sync trip). Marker DGXSTATUS0809.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-status-0809-ts.log
{
echo "DGXSTATUS0809 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSTATUS0809 DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>&1
echo '--- TRAIN/GIT PROC ---'
ps -eo pid,etime,cmd | grep -E '[p]po_train|[t]rain\.py|[c]net2' | grep -v grep | head -5 || echo NO_TRAIN_PROC
ps -eo pid,etime,cmd | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO_GIT_PROC
echo '--- REPO HEAD ---'
cd /home/arm/junyang/carbon-repo && git rev-parse HEAD && timeout 60 git ls-remote origin HEAD | awk '{print "remote:", $1}'
echo '--- IT85 FILES ---'
md5sum code/exp/it85.py data/weights/ckpt_85_frozen.npz 2>&1; ls -l code/exp/it85.py data/weights/ckpt_85_frozen.npz
echo '--- NEW DISPATCH MARKERS (K0/brush/producer) ---'
ls -lt /home/arm/junyang/carbon-repo/data/weights/ 2>/dev/null | head -5
ls /home/arm/junyang/carbon-repo/code/exp/ 2>/dev/null | tail -8
find /home/arm/junyang/carbon-repo -maxdepth 2 -newer /home/arm/junyang/carbon-repo/data/weights/ckpt_85_frozen.npz -name '*.py' 2>/dev/null | head -10
echo '--- SYNC BRANCH HEADS (local refs, expected absent) ---'
git rev-parse sync/h29 sync/h31 2>&1 | head -3
echo "DGXSTATUS0809 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXSTATUS0809 SSH-RC=$?"
echo "DGXSTATUS0809 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
