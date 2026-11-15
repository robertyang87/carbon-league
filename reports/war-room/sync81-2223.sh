#!/bin/bash
# War-room sync trip v81 ~22:23 CST 2026-11-14. READ-ONLY direct reads. Marker SYNC81.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/sync81-2223.log
SSH="timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand=\"nc -X connect -x hatch-egress-proxy:3130 %h %p\""
{
echo "SYNC81 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== L20 gpu-l20-6 100.79.251.87 ==="
eval "$SSH root@100.79.251.87" 'hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null; echo --procs--; ps -eo pid,etime,args | grep -E "[r]l2_ppo|[t]rain\.py" | cut -c1-200 | head -5; echo --examiner--; ls -l /root/carbon/code/exp/master2_src.py /root/carbon/code/exp/echo2.py /root/carbon/data/weights/cnet2_m3w5L.npz 2>&1; md5sum /root/carbon/code/exp/master2_src.py /root/carbon/code/exp/echo2.py /root/carbon/data/weights/cnet2_m3w5L.npz 2>/dev/null; echo --recent--; ls -lt /root/carbon/data/weights 2>/dev/null | head -5; ls -lt /root/carbon/data 2>/dev/null | head -8' 2>&1
echo "L20 SSH-RC=$?"
echo "=== DGX dgx-spark 100.84.148.73 ==="
eval "$SSH arm@100.84.148.73" 'hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null; echo --procs--; ps -eo pid,etime,args | grep -E "[t]rain|[k]0_annotate|[b]uild" | cut -c1-200 | head -5; echo --repo--; git -C /home/arm/junyang/carbon-repo rev-parse HEAD 2>&1; git -C /home/arm/junyang/carbon-repo log --oneline -1 2>&1; echo --returns--; ls /home/arm/junyang/k0_dist/returns/c1 2>/dev/null | wc -l; ls /home/arm/junyang/k0_dist/returns/c2 2>/dev/null | wc -l; ls -lt /home/arm/junyang/k0_dist/returns/c1 2>/dev/null | head -3; ls -lt /home/arm/junyang/k0_dist/returns/c2 2>/dev/null | head -3' 2>&1
echo "DGX SSH-RC=$?"
echo "=== Studio mac-studio 100.78.149.20 ==="
eval "$SSH arm@100.78.149.20" 'hostname; uptime; echo --procs--; ps -eo pid,etime,args | grep -E "[l]auncher|[p]ilot|[q]_gate|[d]u_" | cut -c1-220 | head -8; echo --progress--; find /Users/arm/junyang/carbon -maxdepth 4 -name "*progress*.json" -newermt "2026-11-14 21:00" 2>/dev/null | head; ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -5' 2>&1
echo "Studio SSH-RC=$?"
echo "=== C1 spot-29 100.65.8.27 ==="
eval "$SSH root@100.65.8.27" 'hostname; uptime; cat /proc/loadavg; echo --procs--; ps -eo pid,etime,args | grep -E "[k]0_annotate|[a]nnotate" | cut -c1-220 | head -8; ps -eo pid,etime,args | grep -c "[k]0_annotate" ; echo --out--; ls /root/carbon/data/k0_out* /root/carbon/k0_dist 2>/dev/null; find /root/carbon -maxdepth 4 -name "*shard*" -newermt "2026-11-14 19:00" 2>/dev/null | head -5; find /root/carbon -maxdepth 3 -name "progress*.json" -newermt "2026-11-14 19:00" 2>/dev/null | head -5' 2>&1
echo "C1 SSH-RC=$?"
echo "=== C2 spot-31 100.98.67.20 ==="
eval "$SSH root@100.98.67.20" 'hostname; uptime; cat /proc/loadavg; echo --procs--; ps -eo pid,etime,args | grep -E "[k]0_annotate|[a]nnotate" | cut -c1-220 | head -8; ps -eo pid,etime,args | grep -c "[k]0_annotate"; echo --out--; find /root/carbon -maxdepth 4 -name "*shard*" -newermt "2026-11-14 19:00" 2>/dev/null | head -5; find /root/carbon -maxdepth 3 -name "progress*.json" -newermt "2026-11-14 19:00" 2>/dev/null | head -5' 2>&1
echo "C2 SSH-RC=$?"
echo "SYNC81 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
