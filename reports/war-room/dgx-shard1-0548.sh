#!/bin/bash
# DGX shard1 build outcome check 2026-11-14 ~05:48 (war-room-sync trip). READ-ONLY.
MARK=DGXSHARD10548
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-shard1-0548-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXSHARD10548 DGX-BEGIN $(date '+%F %T %Z')"
D=/home/arm/junyang/carbon/data/ds_gen1r_shard1
echo '--- SHARD1 DIR ---'
ls -la $D 2>&1 | head -20
echo '--- META ---'
cat $D/meta.json 2>&1 | head -c 1500; echo
ls -l $D/*.json 2>/dev/null
echo '--- SHARD COUNT ---'
ls $D/shards/*.npz 2>/dev/null | wc -l
ls -l --time-style=+%F_%T $D/shards/ 2>/dev/null | tail -3
echo '--- BUILD LOGS (recent) ---'
ls -lt /home/arm/junyang/carbon/data/*shard1* /home/arm/junyang/*shard1* /home/arm/junyang/carbon/*shard1* 2>/dev/null | head -10
ls -lt /home/arm/junyang/*.log 2>/dev/null | head -8
echo '--- TOLERANCE CODE IN LOCAL BUILD ---'
grep -n "1e-4\|ulp\|float32" /home/arm/junyang/carbon-repo/code/cnet2_build.py 2>/dev/null | head -20 || grep -rn "1e-4" /home/arm/junyang/carbon/code/cnet2_build.py 2>/dev/null | head -20
echo '--- 994577 ---'
ls $D/shards/ | grep -c 994577; ls $D/shards/ | grep 994577 | head -3
echo "DGXSHARD10548 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
