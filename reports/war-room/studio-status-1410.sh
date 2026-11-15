#!/bin/bash
# Studio status probe 1: user asked 14:10 in Studio sub-chat "跑起来了吗？在干什么任务？"
# Triple check: process + log/weight mtime + log tail (AGENTS.md 2026-11-12 lesson: log tail alone is not proof)
# Marker: STUDIOSTAT1410Q7X9 (claim only output containing this marker)
MARK=STUDIOSTAT1410Q7X9
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1410Q7X9 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
vm_stat | grep -E 'Pages free|Pages active|Pages inactive|Pages wired|Pages occupied|Pages purgeable' ; echo "mem GB total: $(sysctl -n hw.memsize)"; df -h / | tail -1
echo '--- ALL-PYTHON-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'python|train' | grep -v grep || echo NO-PYTHON-PROC
echo '--- CARBON-TREE ---'
ls -d ~/junyang/carbon /Users/arm/junyang/carbon 2>/dev/null; ls ~/junyang/carbon/ 2>/dev/null | head -20
echo '--- DQ-LOGS-NEWEST ---'
ls -lt ~/junyang/carbon/data/dq_logs/ 2>/dev/null | head -15
echo '--- NEW-LOGS-180MIN ---'
find ~/junyang/carbon -maxdepth 4 -name '*.log' -mmin -180 2>/dev/null | head -20
echo '--- WEIGHTS-NEWEST ---'
ls -lt ~/junyang/carbon/data/weights/ 2>/dev/null | head -15
echo '--- DATA-DIRS ---'
du -sh ~/junyang/carbon/data/* 2>/dev/null | head -20
echo '--- DISTILL-DATASET ---'
find ~/junyang/carbon -maxdepth 4 -iname '*distill*' 2>/dev/null | head -20
echo "STUDIOSTAT1410Q7X9 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
