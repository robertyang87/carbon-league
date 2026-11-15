#!/bin/bash
# Hakimi v2 pro TRIP 5, Stage 3 acceptance (READ-ONLY): L20 -> ssh Studio,
# verify the training three-piece on the machine itself:
#   1) training processes registered (cmdline contains hakimi2)
#   2) machine load / per-process CPU
#   3) ds_hakimi2 counts + train-log / weight mtimes fresh
# Internal recheck: if no trainer yet but the chain log shows the build
# still in progress, wait on Studio and re-check (same dispatch, up to
# 3 checks ~150s apart). Reports facts only; exits non-zero only when the
# ssh leg itself fails.
# Trip-5 note: the Studio segment below contains NO inner ssh/rsync, so
# there is no stdin consumer; the outer 'bash -s' heredoc ssh keeps NO -n
# (adding -n would starve the heredoc -- reverse trap).
MARK=HAKIMI2AC5
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 600 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'AEOF'
set -u
MARK=HAKIMI2AC5
D=$HOME/junyang/carbon/data
VENV=$HOME/junyang/carbon/.venv/bin/python
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z')"
hostname
check() {
  echo "--- CHECK $1 $(date '+%F %T %Z') ---"
  echo '[PROCS]'
  pgrep -fl "cnet2_train.py" | cut -c1-220 || echo NO-TRAIN-PROC
  echo '[LOAD]'
  uptime
  echo '[PS-TRAIN]'
  for p in $(pgrep -f "cnet2_train.py"); do ps -o pid,%cpu,%mem,etime,command -p $p | tail -1 | cut -c1-220; done
  echo '[CHAIN-LOG-TAIL]'
  tail -20 $D/hakimi2_chain.log 2>/dev/null || echo NO-CHAIN-LOG
  echo '[DS-META]'
  $VENV -c "import json; d=json.load(open('$D/ds_hakimi2/meta.json')); print('DS-META games=', len(d['games']), 'total=', d['total'], 'teachers=', d['teachers'])" 2>/dev/null || echo NO-DS-META
  echo '[TRAIN-LOGS]'
  for a in r1 g1; do
    echo "== $a =="
    grep -h '^best' $D/hakimi2$a.log 2>/dev/null | tail -2
    tail -2 $D/hakimi2$a.log 2>/dev/null | cut -c1-220
    stat -f 'mtime %Sm %N' -t '%F %T' $D/hakimi2$a.log 2>/dev/null
  done
  echo '[WEIGHTS]'
  ls -lt $D/weights/ 2>/dev/null | grep -i hakimi2 | head -10
  for f in $D/weights/cnet2_hakimi2*_ep*.npz; do
    if [ -f "$f" ]; then echo "MD5 $f $(md5 -q $f)"; fi
  done
}
check 1
N=1
while ! pgrep -f "cnet2_train.py" >/dev/null && [ $N -lt 3 ]; do
  if tail -5 $D/hakimi2_chain.log 2>/dev/null | grep -q "FAIL"; then echo "CHAIN-FAIL-SEEN, stop recheck"; break; fi
  echo "no trainer yet (build may still be running); wait 150s and recheck"
  sleep 150
  N=$((N+1))
  check $N
done
echo "$MARK STUDIO-END $(date '+%F %T %Z')"
AEOF
RC=$?
echo "$MARK L20-END rc=$RC"
if [ $RC -ne 0 ]; then echo "$MARK FAIL ssh-leg rc=$RC"; exit 1; fi
exit 0
