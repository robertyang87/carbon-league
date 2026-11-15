#!/bin/bash
# War-room sync 19:33 probe: L20 value-diag + C1/C2 K0 annotation + DGX archive/returns.
# Marker WARROOMSYNC1933. Read-only direct reads via Tailscale SSH.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-1933-probe.log
{
echo "WARROOMSYNC1933 PROBE-BEGIN $(date '+%F %T %Z')"

echo "=== L20 root@100.79.251.87 ==="
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'echo MARKER-L20VD1933; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[r]un_all|[r]l2_ppo" | head -8; echo ---VD-DIR---; ls -lat /root/ppo_attrib/value-diag/ 2>&1 | head -25; echo ---VD-NEW-FILES---; find /root/ppo_attrib/value-diag -type f -newermt "2026-11-14 19:12" 2>/dev/null | head -20; echo ---END-L20---' 2>&1
echo "L20 SSH-RC=$?"

echo "=== C1 spot29 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot29 'echo MARKER-C1K01933; hostname; uptime; cat /proc/loadavg; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate" | head -4; echo "annotate-count: $(pgrep -f "k0_annotat[e]" | wc -l)"; echo ---OUT---; ls /root/carbon/k0annot/ 2>&1 | head; echo ---PROGRESS---; cat /root/carbon/k0annot/out_timing/progress.json 2>/dev/null | head -c 1200; echo; echo ---SHARDS---; ls /root/carbon/k0annot/out_timing/shards/ 2>/dev/null | wc -l; echo ---END-C1---' 2>&1
echo "C1 SSH-RC=$?"

echo "=== C2 spot31 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot31 'echo MARKER-C2K01933; hostname; uptime; cat /proc/loadavg; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate" | head -4; echo "annotate-count: $(pgrep -f "k0_annotat[e]" | wc -l)"; echo ---OUT---; ls /root/carbon/k0annot/ 2>&1 | head; echo ---PROGRESS---; cat /root/carbon/k0annot/out_timing/progress.json 2>/dev/null | head -c 1200; echo; for d in /root/carbon/k0annot/*/; do echo "$d shards: $(ls "$d"shards/ 2>/dev/null | wc -l)"; done; echo ---END-C2---' 2>&1
echo "C2 SSH-RC=$?"

echo "=== DGX dgx ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config dgx 'echo MARKER-DGX1933; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1; echo ---ARCHIVE---; ls -la /home/arm/junyang/l20_ppo_u_archive_20261114/ 2>&1; find /home/arm/junyang/l20_ppo_u_archive_20261114 -type f 2>/dev/null | wc -l; echo ---RETURNS---; echo "c1:"; ls /home/arm/junyang/k0_dist/returns/c1/ 2>&1 | head -5; echo "c2:"; ls /home/arm/junyang/k0_dist/returns/c2/ 2>&1 | head -5; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate|[r]l2_ppo" | grep -v grep | head -5; echo NONE-ABOVE-MEANS-CLEAN; echo ---END-DGX---' 2>&1
echo "DGX SSH-RC=$?"

echo "WARROOMSYNC1933 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
