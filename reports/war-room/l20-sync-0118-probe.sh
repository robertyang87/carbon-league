#!/bin/bash
# L20 it80 preheat-round check 2026-11-14 ~01:18 (war-room-sync trip).
# Context: it78 (ii) swap executed 23:45:59, PID 30478, it79 row kl 0.0787
# kl_coef 0.15 wr 0.500; it80 est ~01:13 (44-min rounds), it80 EVAL est
# ~01:20-01:30. Pending item: it80 warmup-round calibration error (value MSE
# -> U RMSE) + utility-head sha vs initial_head_sha256 c0261a4f... + actor
# bitwise-unchanged evidence. §19 mitigation (rolling state_last.pt copy
# from it81) approved by war room, not yet due at it80.
# READ-ONLY: no process touches, no file writes on L20.

MARK=L20SYNC0118A1B2
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/l20-sync-0118-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'bash -s' <<'STEOF'
echo "L20SYNC0118A1B2 L20-BEGIN $(date '+%F %T %Z')"
hostname; uptime
D=/root/carbon/data/rl3/snipe_n1
LG=/root/carbon/data/rl3/snipe_n1.log
W=/root/carbon/data/weights
echo '--- TRAIN-PROC ---'
ps -eo pid,etime,args | grep '[r]l2_ppo_train' || echo NO-TRAIN-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>&1
echo '--- TRAIN-JSONL-TAIL-6 ---'
tail -6 $D/train.jsonl 2>&1
echo '--- TRAIN-JSONL-LAST-IT ---'
tail -1 $D/train.jsonl 2>&1 | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print({k:d.get(k) for k in ('it','kl','kl_coef','wr','games','dlogp')})" 2>&1
echo '--- EVAL-JSONL-TAIL-3 ---'
tail -3 $D/eval.jsonl 2>&1
echo '--- LOG-MTIME-SIZE ---'
ls -l $LG; date -r $LG '+%F %T %Z'
echo '--- TRACEBACK-COUNT ---'
grep -c Traceback $LG 2>&1
echo '--- LOG-TAIL-10 ---'
tail -10 $LG
echo '--- STATE-LAST ---'
ls -l $D/state_last.pt $D/state_pre_vhead_it78.pt $D/state_pre_r18_it79.pt 2>&1
echo '--- ROLLING-COPIES ---'
ls -lt $D/ | grep -iE 'state_last.*(copy|bak|\.1|\.2)|copy.*state' || echo NO-ROLLING-COPIES
echo '--- EFFECTIVE-PARAMS ---'
cat $D/effective_params.json 2>&1
echo '--- CKPT-LAST-3 ---'
ls -td $D/ckpt_* 2>/dev/null | head -3 | while read d; do stat -c '%y %n' "$d"; done
echo "L20SYNC0118A1B2 L20-END"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
