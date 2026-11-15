#!/bin/bash
# War-room-sync 09:15 trip: full k0_a0/k0_a1 listing + home-wide npz sweep (today). READ-ONLY. Marker DGXK0A0915.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0a0-0915-ts.log
{
echo "DGXK0A0915 PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK0A0915 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- ALL k0_a0 / k0_a1 PATHS UNDER HOME ---'
ls -la /home/arm/junyang/ | grep -i 'k0' 
find /home/arm/junyang -maxdepth 2 -iname '*k0_a0*' -o -maxdepth 2 -iname '*k0_a1*' 2>/dev/null | head -20
echo '--- HOME-WIDE NPZ TODAY (no maxdepth) ---'
find /home/arm/junyang -name '*.npz' -newermt '2026-11-14 00:00' 2>/dev/null | head -20
echo '--- A0 COMPLETE/LOG IF ANY ---'
cat /home/arm/junyang/k0_a0_out.complete.json 2>/dev/null | head -5
ls -lt /home/arm/junyang/k0_a1_out/ 2>/dev/null; echo "k0_a1_out ls rc=$?"
find /home/arm/junyang/k0_a1_out -type f 2>/dev/null | wc -l
echo '--- ANY .pt/.safetensors TODAY ---'
find /home/arm/junyang -name '*.pt' -o -name '*.safetensors' 2>/dev/null | xargs -r ls -lt 2>/dev/null | head -5
echo "DGXK0A0915 DGX-END $(date '+%F %T %Z')"
STEOF
echo "DGXK0A0915 SSH-RC=$?"
echo "DGXK0A0915 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
