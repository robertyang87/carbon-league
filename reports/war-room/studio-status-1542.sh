#!/bin/bash
# Studio status probe 1542: 15:42 routine check — r1/g1 done, Fable5.2-Max-V5 dual-arm dispatched 15:21, Studio on standby awaiting dataset-ready signal; war-room probe r1fable-1533 ETA ~15:42; check still idle + whether Fable dataset has landed on Studio
# Marker: STUDIOSTAT1542XQ7M (claim only output containing this marker)
MARK=STUDIOSTAT1542XQ7M
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1542XQ7M STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- LOAD ---'
uptime
echo '--- FABLE-DATASET ---'
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -20
find /Users/arm/junyang/carbon/data -iname '*fable*' -maxdepth 3 2>/dev/null | head -20; echo FABLE-FIND-DONE
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -8
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1542XQ7M STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
