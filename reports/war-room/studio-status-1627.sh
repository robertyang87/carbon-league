#!/bin/bash
# Studio status probe 1627: 16:27 routine check — war room confirmed ~16:26 it has started pulling ds_fable (DGX -> Studio via LAN 10.188.0.23) and starting Fable dual-arm (cnet2_fabler1 / cnet2_fableg1); check whether ds_fable has landed and whether training has started (three-piece: process + MPS/load + log/weight mtime)
# Marker: STUDIOSTAT1627R8K2 (claim only output containing this marker)
MARK=STUDIOSTAT1627R8K2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1627R8K2 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|fable' | grep -v grep || echo NO-TRAIN-PROC
echo '--- FABLE-DATASET ---'
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -25
find /Users/arm/junyang/carbon/data -iname '*fable*' -maxdepth 3 2>/dev/null | head -20; echo FABLE-FIND-DONE
echo '--- DS_FABLE-CONTENT ---'
ls /Users/arm/junyang/carbon/data/ds_fable 2>/dev/null | head -5; du -sh /Users/arm/junyang/carbon/data/ds_fable 2>/dev/null; find /Users/arm/junyang/carbon/data/ds_fable -type f 2>/dev/null | wc -l
echo '--- WEIGHTS-NOW ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | head -10
echo '--- FABLE-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
find /Users/arm/junyang/carbon/data -iname '*fable*.log' -maxdepth 2 2>/dev/null | while read -r f; do echo "== $f =="; tail -5 "$f"; done; echo FABLE-LOGS-DONE
echo '--- DATA-TOTAL ---'
du -sh /Users/arm/junyang/carbon/data 2>/dev/null
echo "STUDIOSTAT1627R8K2 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
