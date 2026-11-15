#!/bin/bash
# War-room-sync 10:33 trip: read-only fleet probe. Marker SYNC1033.
MARK=SYNC1033
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/sync-1033-probe.log
SSH="ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand=nc\ -X\ connect\ -x\ hatch-egress-proxy:3130\ %h\ %p"
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
echo '=== L20 root@100.79.251.87 ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'echo ---TRAIN---; tail -2 /root/carbon/data/rl2/train.jsonl 2>/dev/null; echo ---PID---; ps -eo pid,etime,args | grep "[r]l2_ppo_train" | cut -c1-200; echo ---GPU---; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv 2>&1 | tail -1; echo ---SPAWN---; pgrep -f "spawn_mai[n]" | wc -l; echo ---STATE---; ls -l --time-style=+%H:%M:%S /root/carbon/data/weights/state_last.pt 2>/dev/null; ls /root/carbon/data/weights/ckpt_9*.npz 2>/dev/null | tail -3' 2>&1
echo "L20 SSH-RC=$?"
echo '=== DGX arm@100.84.148.73 ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'echo ---HEAD---; git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD 2>&1; echo ---STATUS---; git -C /home/arm/junyang/carbon-repo status --porcelain 2>&1 | wc -l; echo ---SCREEN---; ps -eo pid,etime,args | grep "[s]creen" | head -5 || echo NO-SCREEN; echo ---K0TRAIN---; md5sum /home/arm/junyang/carbon-repo/code/exp/k0_train*.py /home/arm/junyang/carbon/code/exp/k0_train*.py 2>&1; echo ---CERTIFY---; md5sum /home/arm/junyang/carbon-repo/code/certify*.py /home/arm/junyang/carbon/code/certify*.py 2>&1; echo ---CONFLICTLIST---; ls -l /tmp/s1_conflict_list.txt 2>&1; echo ---BACKUPDIR---; ls -d /home/arm/junyang/*backup* /home/arm/junyang/carbon-repo-backup* 2>&1 | head; echo ---GPU---; nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv 2>&1 | tail -1' 2>&1
echo "DGX SSH-RC=$?"
echo '=== Studio arm@100.78.149.20 ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'echo ---PILOT---; ps -eo pid,etime,args | grep -E "[p]1_launch|[e]ndgame" | cut -c1-200 | head -5; echo ---WORKERS---; ps -eo pid,etime,%cpu,args | grep "[p]ilot_worker" | wc -l; echo ---LOAD---; uptime; echo ---P1B---; ls -l --time-style=+%H:%M:%S /Users/arm/junyang/carbon/data/stage1_formal/endgame_p1b.log /Users/arm/junyang/carbon/data/stage1_formal/endgame_p1a.json.pilot.json 2>&1; echo ---GATE---; ps -eo pid,etime,args | grep "[g]ate 59060" | cut -c1-150 | head -2' 2>&1
echo "STUDIO SSH-RC=$?"
echo '=== C1 spot-29 root@100.65.8.27 ==='
timeout -k 30 90 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.65.8.27 'hostname; uptime; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[b]igeval" | head -3 || echo NO-EXAM-PROC; ls /root/carbon/data/bigeval_live 2>/dev/null | wc -l' 2>&1
echo "C1 SSH-RC=$?"
echo '=== C2 spot-31 root@100.98.67.20 ==='
timeout -k 30 90 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.98.67.20 'hostname; uptime; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[b]igeval" | head -3 || echo NO-EXAM-PROC; ls /root/carbon/data/bigeval_live 2>/dev/null | wc -l' 2>&1
echo "C2 SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
