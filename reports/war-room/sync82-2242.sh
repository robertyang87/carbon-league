#!/bin/bash
# War-room sync trip v82 ~22:42 CST 2026-11-14. READ-ONLY direct reads. Marker SYNC82.
# Direct ssh only (no eval) - see AGENTS.md 2026-11-14 lesson.
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/sync82-2242.log
SSHX() { timeout -k 30 240 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" "$@"; }
{
echo "SYNC82 PROBE-BEGIN $(date '+%F %T %Z')"
echo "=== L20 gpu-l20-6 100.79.251.87 ==="
SSHX root@100.79.251.87 'hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1; echo --bigeval-procs--; ps -eo pid,etime,args | grep "[b]igeval" | cut -c1-200 | head -8; echo --bigeval-count--; ps -eo args | grep -c "[b]igeval"; echo --step0-live--; ls -lt /root/carbon/data/bigeval_live/ 2>/dev/null | head -10; find /root/carbon -maxdepth 4 -name "*ppou2*" 2>/dev/null | head -10; echo --step0-wc--; for f in $(find /root/carbon -maxdepth 4 -name "*ppou2*" -name "*.jsonl" 2>/dev/null | head -6); do echo -n "$f: "; wc -l < "$f" 2>/dev/null; done; echo --probe-hung--; ps -eo pid,etime,args | grep "[r]un_step2b" | cut -c1-160 | head -3; find /root/carbon /root/ppo_attrib -maxdepth 3 -name "step2_done.txt" 2>/dev/null | head -3' 2>&1
echo "L20 SSH-RC=$?"
echo "=== DGX dgx-spark 100.84.148.73 ==="
SSHX arm@100.84.148.73 'hostname; uptime; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>&1; echo --procs--; ps -eo pid,etime,args | grep -E "[t]rain|[k]0_annotate|[b]uild" | cut -c1-200 | head -5; echo --repo--; git -C /home/arm/junyang/carbon-repo rev-parse HEAD 2>&1; git -C /home/arm/junyang/carbon-repo log --oneline -1 2>&1; echo --returns--; echo -n "c1 files: "; ls /home/arm/junyang/k0_dist/returns/c1 2>/dev/null | wc -l; echo -n "c2 files: "; ls /home/arm/junyang/k0_dist/returns/c2 2>/dev/null | wc -l; echo --c1wave2--; cat /home/arm/junyang/k0_dist/returns/c1/progress_c1_wave2.json 2>/dev/null | head -c 400; echo; echo --c2progress--; tail -c 400 /home/arm/junyang/k0_dist/returns/c2/progress.json 2>/dev/null; echo; ls -lt /home/arm/junyang/k0_dist/returns/c1 2>/dev/null | head -3; ls -lt /home/arm/junyang/k0_dist/returns/c2 2>/dev/null | head -3' 2>&1
echo "DGX SSH-RC=$?"
echo "=== Studio mac-studio 100.78.149.20 ==="
SSHX arm@100.78.149.20 'hostname; uptime; echo --procs--; ps -eo pid,etime,args | grep -E "[l]auncher|[s]tage1_teacher_gate" | cut -c1-220 | head -6; echo --qprogress--; find /Users/arm/junyang/carbon -maxdepth 5 -name "*progress*.json" -newermt "2026-11-14 21:00" 2>/dev/null | head -5; echo --qfile--; for f in $(find /Users/arm/junyang/carbon -maxdepth 5 -name "*progress*.json" -newermt "2026-11-14 21:00" 2>/dev/null | head -3); do echo "== $f"; tail -c 500 "$f" 2>/dev/null; echo; done' 2>&1
echo "Studio SSH-RC=$?"
echo "=== C1 spot-29 100.65.8.27 ==="
SSHX root@100.65.8.27 'hostname; uptime; cat /proc/loadavg; echo --procs--; ps -eo pid,etime,args | grep "[k]0_annotate" | cut -c1-160 | head -3; echo --count--; ps -eo args | grep -c "[k]0_annotate"; echo --wave2shards--; ls /root/carbon/k0annot/out_wave2/shards 2>/dev/null | wc -l; ls -lt /root/carbon/k0annot/out_wave2/shards 2>/dev/null | head -3; echo --wave2progress--; cat /root/carbon/k0annot/out_wave2/progress.json 2>/dev/null | head -c 500; echo' 2>&1
echo "C1 SSH-RC=$?"
echo "=== C2 spot-31 100.98.67.20 ==="
SSHX root@100.98.67.20 'hostname; uptime; cat /proc/loadavg; echo --procs--; ps -eo pid,etime,args | grep "[k]0_annotate" | cut -c1-160 | head -3; echo --count--; ps -eo args | grep -c "[k]0_annotate"; echo --wave2shards--; ls /root/carbon/data/k0_annot_c2/shards 2>/dev/null | wc -l; ls -lt /root/carbon/data/k0_annot_c2/shards 2>/dev/null | head -3; echo --progress--; cat /root/carbon/data/k0_annot_c2/progress.json 2>/dev/null | head -c 500; echo' 2>&1
echo "C2 SSH-RC=$?"
echo "SYNC82 PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
