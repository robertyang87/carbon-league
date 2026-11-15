#!/bin/bash
# War-room sync 21:34 probe: L20 it75/U-line + C1 wr1 exam + C2 K0 + DGX returns/archive + Studio run.
# Marker WARROOMSYNC2134. Read-only direct reads via Tailscale SSH.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/war-room-sync-2134-probe.log
{
echo "WARROOMSYNC2134 PROBE-BEGIN $(date '+%F %T %Z')"

echo "=== L20 root@100.79.251.87 ==="
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" root@100.79.251.87 'echo MARKER-L20-2134; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[r]sync|[s]cp" | head -8; echo ---PID28238---; ps -p 28238 -o pid,etime,args 2>&1; echo ---TRAIN-JSONL---; ls -la /root/carbon/data/rl3/train.jsonl 2>&1; tail -3 /root/carbon/data/rl3/train.jsonl 2>/dev/null | cut -c1-400; echo ---STATE---; ls -la /root/carbon/data/rl3/state_pre_termU_it74.pt 2>&1; md5sum /root/carbon/data/rl3/state_pre_termU_it74.pt 2>&1; echo ---END-L20---' 2>&1
echo "L20 SSH-RC=$?"

echo "=== C1 spot29 wr1-exam + K0 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot29 'echo MARKER-C1-2134; hostname; uptime; cat /proc/loadavg; echo ---EXAM-PROCS---; ps -eo pid,etime,args | grep -E "[b]igeval|[e]val_v53|[k]0_annotate" | head -8; echo ---K0-SHARDS---; ls /root/carbon/k0annot/out_timing/shards/ 2>/dev/null | wc -l; cat /root/carbon/k0annot/out_timing/progress.json 2>/dev/null | head -c 400; echo; echo ---WR1-FILES---; ls -lat /root/carbon/data/ 2>/dev/null | grep -iE "wr1|m9xj570|w_" | head -8; find /root/carbon/data -maxdepth 2 -newermt "2026-11-14 21:00" -type f 2>/dev/null | head -12; echo ---END-C1---' 2>&1
echo "C1 SSH-RC=$?"

echo "=== C2 spot31 K0 ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config spot31 'echo MARKER-C2-2134; hostname; uptime; cat /proc/loadavg; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate" | head -3; echo "annotate-count: $(pgrep -f "k0_annotat[e]" | wc -l)"; echo ---SHARDS---; ls /root/carbon/k0annot/out_timing/shards/ 2>/dev/null | wc -l; cat /root/carbon/k0annot/out_timing/progress.json 2>/dev/null | head -c 400; echo; echo ---END-C2---' 2>&1
echo "C2 SSH-RC=$?"

echo "=== DGX dgx ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config dgx 'echo MARKER-DGX2134; hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1; echo ---RETURNS---; ls -la /home/arm/junyang/k0_dist/returns/c1/ 2>&1 | head -6; ls -la /home/arm/junyang/k0_dist/returns/c2/ 2>&1 | head -6; echo ---ARCHIVE---; find /home/arm/junyang/l20_ppo_u_archive_20261114 -type f 2>/dev/null | wc -l; du -sb /home/arm/junyang/l20_ppo_u_archive_20261114 2>/dev/null; ls /home/arm/junyang/l20_ppo_u_archive_20261114/*.md5 /home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST* 2>&1 | head; echo ---REPO---; cd /home/arm/junyang/carbon-repo 2>/dev/null && git log --oneline -3 2>&1; git log --oneline -3 2>&1 | head -3; echo ---DSGEN1---; ls -la /home/arm/junyang/carbon-repo/data/ds_gen1* 2>/dev/null | head -4; find /home/arm/junyang/carbon-repo -maxdepth 3 -name "*ds_gen1*" 2>/dev/null | head -5; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[k]0_annotate|[r]l2_ppo|[s]tage0|[s]elfplay" | grep -v grep | head -4; echo NONE-ABOVE-MEANS-CLEAN; echo ---END-DGX---' 2>&1
echo "DGX SSH-RC=$?"

echo "=== Studio mac-studio ==="
timeout -k 30 180 ssh -F /home/hatch/.ssh/carbon_config mac-studio 'echo MARKER-STUDIO2134; hostname; uptime; echo ---PROCS---; ps -eo pid,etime,args | grep -E "[s]tage1|[g]ate|[p]ilot" | head -6; echo ---QPROGRESS---; ls -lat /Users/arm/junyang/carbon/data/stageB_q/ 2>/dev/null | head -6; find /Users/arm/junyang/carbon -maxdepth 4 -name "*.progress.json" 2>/dev/null | head -5; for f in $(find /Users/arm/junyang/carbon -maxdepth 4 -name "*.progress.json" 2>/dev/null | head -3); do echo "== $f"; cat "$f" 2>/dev/null | head -c 400; echo; done; echo ---END-STUDIO---' 2>&1
echo "Studio SSH-RC=$?"

echo "WARROOMSYNC2134 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
