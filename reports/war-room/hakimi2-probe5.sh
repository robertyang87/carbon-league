#!/bin/bash
# Hakimi v2 pro TRIP 5, Stage 1 probe (SPARE -- trip-5 dispatches Stage 2
# directly; this probe exists only if a channel precheck is needed).
# L20 -> ssh Studio connectivity precheck only. Small on purpose (b64 < 4KB)
# so output lands inside the retrievable log head. Any failure: echo FAIL +
# stderr tail, exit non-zero (workflow must go failure).
MARK=HAKIMI2P5
echo "$MARK BEGIN $(date '+%F %T %Z')"
ERR=/tmp/hakimi2p5.err
OUT=$(timeout -k 30 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'echo STUDIO-REACHABLE; hostname; ls -l ~/junyang/carbon/.venv/bin/python; echo ---LAYOUT---; ls ~/junyang' 2>$ERR)
RC=$?
if [ $RC -ne 0 ]; then
  echo "$MARK FAIL ssh-leg rc=$RC"
  echo "--- ssh stderr tail ---"
  tail -5 $ERR
  echo "$MARK END"
  exit 1
fi
echo "$OUT"
echo "$OUT" | grep -q STUDIO-REACHABLE || { echo "$MARK FAIL no-reachable-marker"; echo "$MARK END"; exit 1; }
echo "$OUT" | grep -q "junyang/carbon/.venv/bin/python" || { echo "$MARK FAIL no-venv-python"; echo "$MARK END"; exit 1; }
echo "$MARK OK"
echo "$MARK END"
exit 0
