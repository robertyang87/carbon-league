#!/bin/bash
# Studio status report 04:12 trip — READ-ONLY snapshot.
# L20 -> ssh Studio, single check, no wait loop, no training launch.
MARK=STUDIO0412
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'AEOF'
set -u
MARK=STUDIO0412
D=$HOME/junyang/carbon/data
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
hostname
uptime
echo '[MEM]'
vm_stat | head -8
echo '[DISK]'
df -h / | tail -1
echo '[TRAIN-PROCS]'
pgrep -fl "cnet2_train.py" | cut -c1-220 || echo NO-TRAIN-PROC
echo '[ANY-PYTHON]'
pgrep -fl python3 | grep -v -E "pgrep|grep" | cut -c1-220 || echo NO-PYTHON
echo '[HAKIMI2-LOGS]'
for a in r1 g1; do
  echo "== $a =="
  grep -h '^best' $D/hakimi2$a.log 2>/dev/null | tail -1
  tail -1 $D/hakimi2$a.log 2>/dev/null | cut -c1-220
  stat -f 'mtime %Sm %N' -t '%F %T' $D/hakimi2$a.log 2>/dev/null || echo NO-LOG-$a
done
echo '[SHIP-LOG]'
tail -5 $D/hakimi2_ship.log 2>/dev/null | cut -c1-220 || echo NO-SHIP-LOG
echo '[WEIGHTS-HAKIMI2]'
ls -lt $D/weights/ 2>/dev/null | grep -i hakimi2 | head -6
echo '[DS-META]'
ls -l $D/ds_hakimi2/meta.json 2>/dev/null || echo NO-DS-HAKIMI2
echo "$MARK STUDIO-END $(date '+%F %T %Z')"
AEOF
RC=$?
echo "$MARK L20-END rc=$RC"
if [ $RC -ne 0 ]; then echo "$MARK FAIL ssh-leg rc=$RC"; exit 1; fi
exit 0
