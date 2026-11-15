#!/bin/bash
# War-room sync 21:34 follow-up probe: L20 paths, C1 wr1 staging, C2 shards (correct path), Studio retry.
# Marker WARROOMSYNC2134B. Read-only.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-2134-probe2.log
{
echo "WARROOMSYNC2134B PROBE-BEGIN $(date '+%F %T %Z')"

echo "=== L20 follow-up ==="
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'echo MARKER-L20B; ls /root/carbon/data/ 2>&1 | head -20; echo ---FIND-TRAIN---; find /root -maxdepth 4 \( -name "train.jsonl" -o -name "snipe_n1.log" -o -name "state_pre_termU_it74.pt" \) 2>/dev/null | head; echo ---ALL-PYTHON---; ps -eo pid,etime,args | grep -E "[p]ython|[r]sync|[s]cp|[r]l2" | head -10; echo ---RL3---; ls -la /root/carbon/data/rl3/ 2>&1 | head -12; echo ---END-L20B---' 2>&1
echo "L20B SSH-RC=$?"

echo "=== C1 wr1 staging follow-up ==="
timeout -k 30 120 ssh -F /home/hatch/.ssh/carbon_config spot29 'echo MARKER-C1B; echo ---ALL-PYTHON---; ps -eo pid,etime,args | grep -E "[p]ython|[b]igeval|[r]sync|[g]it" | grep -v k0_annotate | head -10; echo ---NEWFILES---; find /root/carbon /root -maxdepth 3 -newermt "2026-11-14 21:00" -type f 2>/dev/null | grep -v k0annot | head -15; echo ---WR1PACK---; find /root -maxdepth 4 \( -name "*m9xj570wr1*" -o -name "*570wr1*" \) 2>/dev/null | head; echo ---EXAMDIR---; ls /root/carbon/exp/ 2>/dev/null | tail -8; echo ---SNAPLOOP---; pgrep -af "snaploop|sync.*h29" | head -5; echo ---END-C1B---' 2>&1
echo "C1B SSH-RC=$?"

echo "=== C2 shards correct path ==="
timeout -k 30 120 ssh -F /home/hatch/.ssh/carbon_config spot31 'echo MARKER-C2B; ls /root/carbon/data/k0_annot_c2/shards/ 2>/dev/null | wc -l; ls -la /root/carbon/data/k0_annot_c2/ 2>/dev/null | head -6; echo ---END-C2B---' 2>&1
echo "C2B SSH-RC=$?"

echo "=== Studio retry ==="
timeout -k 30 120 ssh -F /home/hatch/.ssh/carbon_config mac-studio 'echo MARKER-STUDIO2134B; hostname; ps -eo pid,etime,args | grep -E "[s]tage1|[g]ate" | head -4; find /Users/arm/junyang/carbon -maxdepth 4 -name "*.progress.json" 2>/dev/null | head -5; echo ---END-STUDIOB---' 2>&1
echo "StudioB SSH-RC=$?"

echo "WARROOMSYNC2134B PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
