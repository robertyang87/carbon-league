#!/bin/bash
# Phase C trip 2b: benchmark launch, fixed probe (bracket pattern avoids
# pgrep self-match) + snapshot preflight + snapshot #1 to DGX.
MARK=S30CT2C3H7
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 900 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30CT2C3H7
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
echo "--- SNAPSHOT PREFLIGHT ---"
timeout -k 30 60 $SCPCMD root@"$TGT":/root/carbon/data/s30deploy_smoke.out "$SNAP/preflight_smoke.out" </dev/null 2>&1
if [ -s "$SNAP/preflight_smoke.out" ]; then echo "SNAP-PREFLIGHT-OK bytes=$(stat -c%s "$SNAP/preflight_smoke.out")"; else echo "SNAP-PREFLIGHT-FAIL"; echo "$MARK DGX-END"; exit 1; fi
echo "--- LAUNCH BENCHMARK ---"
RUNNING=$(timeout -k 30 60 $SSHCMD root@"$TGT" "pgrep -af \"$PAT\" | head -2" 2>&1)
if [ -n "$RUNNING" ]; then
  echo "BENCH-ALREADY-RUNNING: $RUNNING"
else
  timeout -k 30 60 $SSHCMD root@"$TGT" 'cd /root/carbon/code && nohup env EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ../.venv/bin/python bigeval.py exp/m3s2r1.py exp/opp_master2r.py 996000 200 s30bench > /root/carbon/data/s30bench.out 2>&1 & echo BENCH-LAUNCHED'
fi
sleep 100
echo "--- SNAPSHOT 1 (after ~100s) ---"
LIVE=$(timeout -k 30 60 $SSHCMD root@"$TGT" 'ls /root/carbon/data/bigeval_live/s30bench.jsonl 2>/dev/null || find /root/carbon/data -name "s30bench*.jsonl" 2>/dev/null | head -1' 2>&1)
echo "LIVE-FILE=$LIVE"
if [ -n "$LIVE" ]; then
  timeout -k 30 120 $SCPCMD root@"$TGT":"$LIVE" "$SNAP/s30bench.jsonl" </dev/null 2>&1
  echo "SNAP1-LINES=$(wc -l < "$SNAP/s30bench.jsonl")"
else
  echo "SNAP1-NO-LIVE-FILE-YET"
fi
ALIVE=$(timeout -k 30 60 $SSHCMD root@"$TGT" "pgrep -af \"$PAT\" | head -1" 2>&1)
echo "BENCH-ALIVE: $ALIVE"
timeout -k 30 60 $SSHCMD root@"$TGT" 'tail -2 /root/carbon/data/s30bench.out' 2>&1
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
