#!/bin/bash
# Probe C-recon: diff repo vs workspace code, read play_batch body, bench harness, term/dworth fields.
# Read-only. Marker: PBRECON2319C
MARK=PBRECON2319C
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "PBRECON2319C DGX-BEGIN $(date '+%F %T %Z')"
R=/home/arm/junyang/carbon-repo/code
W=/home/arm/junyang/carbon/code
echo '--- DIFF NAMES repo vs workspace (py only) ---'
diff -rq $R $W 2>/dev/null | grep -E '\.py' | head -30
echo '--- ROLLOUT DIFF (repo=new vs workspace=old) ---'
diff -u $W/rl2_ppo_rollout.py $R/rl2_ppo_rollout.py | head -160
echo '--- BENCH HARNESS ---'
cat $R/rl2_ppo_bench.py
echo '--- PLAY_BATCH BODY 155-340 ---'
sed -n '155,340p' $R/rl2_ppo_rollout.py
echo '--- TERM/DWORTH GREP (rollout) ---'
grep -n "dworth\|extinct\|term" $R/rl2_ppo_rollout.py | head -40
echo '--- TERM GREP (eval_v53) ---'
grep -n "extinct\|term" $R/eval_v53.py | head -30
echo "PBRECON2319C DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
