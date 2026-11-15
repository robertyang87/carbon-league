#!/bin/bash
# War-room sync 21:03 probe: L20 archive + C1/C2 K0 annotation + DGX returns + Studio q gate.
# Marker WARROOMSYNC2103. Read-only direct reads via Tailscale SSH.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-2103-probe.log
{
echo "WARROOMSYNC2103 PROBE-BEGIN $(date '+%F %T %Z')"

echo "=== L20 root@100.79.251.87 ==="
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'echo MARKER-L20-2103; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[r]un_all" | head -6; echo ---TRAIN-JSONL---; ls -la /root/carbon/data/rl3/train.jsonl 2>&1; tail -2 /root/carbon/data/rl3/train.jsonl 2>/dev/null | cut -c1-300; echo ---ARCHIVE---; ls /home/arm/junyang/l20_ppo_u_archive_20261114/ 2>/dev/null; find /root/ppo_attrib -maxdepth 2 -type d 2>/dev/null | head; echo ---END-L20---' 2>&1
echo "L20 SSH-RC=$?"

echo "=== C1 spot29 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot29 'echo MARKER-C1K02103; hostname; uptime; cat /proc/loadavg; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate" | head -3; echo "annotate-count: $(pgrep -f "k0_annotat[e]" | wc -l)"; echo ---OUT---; ls /root/carbon/data/ 2>/dev/null | grep -i k0; echo ---SHARDS-C1---; ls /root/carbon/data/k0_annot_c1/shards/ 2>/dev/null | wc -l; ls -la /root/carbon/data/k0_annot_c1/ 2>/dev/null | head -8; echo ---PROGRESS---; cat /root/carbon/data/k0_annot_c1/progress.json 2>/dev/null | head -c 800; echo; echo ---END-C1---' 2>&1
echo "C1 SSH-RC=$?"

echo "=== C2 spot31 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot31 'echo MARKER-C2K02103; hostname; uptime; cat /proc/loadavg; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate" | head -3; echo "annotate-count: $(pgrep -f "k0_annotat[e]" | wc -l)"; echo ---SHARDS-C2---; ls /root/carbon/data/k0_annot_c2/shards/ 2>/dev/null | wc -l; ls -la /root/carbon/data/k0_annot_c2/ 2>/dev/null | head -8; echo ---PROGRESS---; cat /root/carbon/data/k0_annot_c2/progress.json 2>/dev/null | head -c 800; echo; echo ---END-C2---' 2>&1
echo "C2 SSH-RC=$?"

echo "=== DGX dgx ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config dgx 'echo MARKER-DGX2103; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1; echo ---RETURNS---; echo "c1:"; ls -la /home/arm/junyang/k0_dist/returns/c1/ 2>&1 | head -8; echo "c2:"; ls -la /home/arm/junyang/k0_dist/returns/c2/ 2>&1 | head -8; echo ---ARCHIVE---; find /home/arm/junyang/l20_ppo_u_archive_20261114 -type f 2>/dev/null | wc -l; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate|[r]l2_ppo" | grep -v grep | head -4; echo NONE-ABOVE-MEANS-CLEAN; echo ---END-DGX---' 2>&1
echo "DGX SSH-RC=$?"

echo "=== Studio mac-studio ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config mac-studio 'echo MARKER-STUDIO2103; hostname; uptime; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[s]tage1|[q]_gate|[g]ate" | head -5; echo ---QPROGRESS---; ls -lat /Users/arm/junyang/carbon/data/stageB_q/ 2>/dev/null | head -8; cat /Users/arm/junyang/carbon/data/stageB_q/*.progress.json 2>/dev/null | head -c 600; echo; echo ---END-STUDIO---' 2>&1
echo "Studio SSH-RC=$?"

echo "WARROOMSYNC2103 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
