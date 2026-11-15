#!/bin/bash
# Phase C trip 3 (final): poll on DGX side up to ~6 min; each check refreshes
# the snapshot (overwrite) on DGX ~/s30bench_snapshots/; stop when the
# s30bench summary line parses with n==200, then print final stats.
MARK=S30CT3C9D5
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 900 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30CT3C9D5
TGT=47.117.116.55
SNAP=$HOME/s30bench_snapshots
mkdir -p "$SNAP"
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@"$TGT" 'echo SSH-OK $(hostname)' 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK SSH-NOT-READY"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
SSHCMD="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
SCPCMD="scp -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
PAT='bigeval[.]py.*s30bench'
DONE=0
for i in 1 2 3 4 5 6 7 8; do
  LIVE=$(timeout -k 30 60 $SSHCMD root@"$TGT" 'ls /root/carbon/data/bigeval_live/s30bench.jsonl 2>/dev/null || find /root/carbon/data -name "s30bench*.jsonl" 2>/dev/null | head -1' 2>&1)
  if [ -n "$LIVE" ]; then
    timeout -k 30 120 $SCPCMD root@"$TGT":"$LIVE" "$SNAP/s30bench.jsonl" </dev/null 2>&1
    echo "CHECK$i SNAP-LINES=$(wc -l < "$SNAP/s30bench.jsonl" 2>/dev/null)"
  else
    echo "CHECK$i NO-LIVE-FILE"
  fi
  SUMLINE=$(timeout -k 30 60 $SSHCMD root@"$TGT" 'grep -a "\"tag\": \"s30bench\"" /root/carbon/data/s30bench.out | tail -1' 2>&1)
  if echo "$SUMLINE" | grep -q '"n": 200'; then
    echo "BENCH-SUMMARY-LINE: $SUMLINE"
    DONE=1
    break
  fi
  ALIVE=$(timeout -k 30 60 $SSHCMD root@"$TGT" "pgrep -af \"$PAT\" | head -1" 2>&1)
  echo "CHECK$i ALIVE: $ALIVE"
  if [ -z "$ALIVE" ]; then echo "CHECK$i PROCESS-GONE"; break; fi
  sleep 45
done
if [ "$DONE" = "1" ]; then
  echo "SNAP-FINAL lines=$(wc -l < "$SNAP/s30bench.jsonl") md5=$(md5sum "$SNAP/s30bench.jsonl" | awk '{print $1}') bytes=$(stat -c%s "$SNAP/s30bench.jsonl") path=$SNAP/s30bench.jsonl"
  echo "$SUMLINE" | python3 -c '
import sys, json
line = sys.stdin.read().strip()
d = json.loads(line[line.index("{"):])
print("BENCH-PARSED " + json.dumps(d, sort_keys=True))
'
else
  echo "BENCH-NOT-DONE-IN-WINDOW"
  timeout -k 30 60 $SSHCMD root@"$TGT" 'tail -3 /root/carbon/data/s30bench.out' 2>&1
fi
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
