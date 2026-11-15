#!/bin/bash
MARK=WG1STATE1925C
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "WG1STATE1925C STUDIO-BEGIN $(date '+%F %T %Z')"
D=$HOME/junyang/carbon/data
echo '--- TRAIN-PROC ---'
pgrep -fl "cnet2_train.py" || echo NO-TRAIN-PROC
echo "WG1_PIDFILE=$(cat $D/m3s2wg1.pid 2>/dev/null)"
ps -o pid,etime,%cpu,rss,command -p $(cat $D/m3s2wg1.pid 2>/dev/null) 2>&1 | cut -c1-200
echo '--- SHIP-WATCHER ---'
pgrep -fl "ship_m3s2wg1" || echo NO-SHIP-WATCHER
echo '--- WG1-LOG ---'
ls -l --time-style=+%F_%T $D/m3s2wg1.log 2>/dev/null || ls -l $D/m3s2wg1.log 2>/dev/null || echo NO-LOG
tail -6 $D/m3s2wg1.log 2>/dev/null
echo '--- WEIGHTS ---'
ls -lt $D/weights/ | grep -i m3s2wg1 | head -5
echo "WG1STATE1925C STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
