#!/bin/bash
# War-room-sync 10:33 trip follow-up: close L20 path gap + DGX k0_train/backup + Studio p1b. Marker SYNC1033B.
MARK=SYNC1033B
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/sync-1033-probe2.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
echo '=== L20 paths ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'find /root/carbon -maxdepth 4 -name "train.jsonl" 2>/dev/null; find /root/carbon -maxdepth 4 -name "state_last.pt" 2>/dev/null; ls -l /root/carbon/data/ 2>/dev/null | head -20' 2>&1
echo "L20-FIND RC=$?"
echo '=== L20 train tail (path from find) ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'for f in $(find /root/carbon -maxdepth 4 -name "train.jsonl" 2>/dev/null); do echo "-- $f"; tail -2 "$f"; done; for f in $(find /root/carbon -maxdepth 4 -name "state_last.pt" 2>/dev/null); do echo "-- $f"; ls -l --time-style=+%H:%M:%S "$f"; done; ls /root/carbon/data/weights/ 2>/dev/null | grep -E "ckpt_8[0-9]|ckpt_9[0-9]" | tail -4' 2>&1
echo "L20-TAIL RC=$?"
echo '=== DGX k0_train + backup + status ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -c "find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -name \"k0_train*\" 2>/dev/null; echo ---MD5---; find /home/arm/junyang/carbon-repo /home/arm/junyang/carbon -name \"k0_train*\" -exec md5sum {} + 2>/dev/null; echo ---STATUS---; git -C /home/arm/junyang/carbon-repo status --porcelain 2>&1; echo ---NEWDIRS---; find /home/arm/junyang -maxdepth 2 -newermt \"2026-11-14 10:30\" -type d 2>/dev/null | head; echo ---CONFLICT-REMAIN---; ls /home/arm/junyang/carbon-repo/code/exp/k0_a0.py /home/arm/junyang/carbon-repo/data/fable6_corpus 2>&1 | head -3; echo ---SCREENPROC---; ps -eo pid,etime,args | grep screen | grep -v grep | head -3; echo NONE-IF-EMPTY"' 2>&1
echo "DGX RC=$?"
echo '=== Studio p1b + workers ==='
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=25 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.78.149.20 'ls -lT /Users/arm/junyang/carbon/data/stage1_formal/endgame_p1b.log /Users/arm/junyang/carbon/data/stage1_formal/endgame_p1a.json.pilot.json 2>&1; echo ---PYPROCS---; pgrep -f python | wc -l; ps -eo pid,etime,%cpu,args | grep "stage1_teacher_gate" | grep -v grep | wc -l; echo ---LOAD---; uptime' 2>&1
echo "STUDIO RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
