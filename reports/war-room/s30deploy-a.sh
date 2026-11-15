#!/bin/bash
# Deploy new harness (main 639253c0) — Phase A: DGX prep.
# War-room order 02:21. Marker: S30DEPLOY0225A
# NOTE: s26stage code/ sync is CONDITIONAL — the spot-30 push worker is
# still rsyncing s26stage -> spot-30 (launched 02:07:19); mutating code/
# mid-transfer would corrupt its MD5-COMPARE. Sync runs only if no push
# rsync is alive; otherwise it is DEFERRED and done at the Phase C gate.
MARK=S30DEPLOY0225A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 900 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30DEPLOY0225A
R=/home/arm/junyang/carbon-repo
ST=/home/arm/junyang/s26stage
OLD=0f704a9da8da05e68976bfd1442467b52106bf5c
FILES="code/eval_v53.py code/bigeval.py code/cnet2_search.py code/cnet2_build.py code/cnet2_search_timing.py code/replica_chain.sh code/rl2_ppo_rollout.py code/cnet2_np.py"
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo "--- HEAD-BEFORE ---"
git -C "$R" rev-parse HEAD
git -C "$R" branch --show-current
echo "--- FETCH+FF ---"
timeout 120 git -C "$R" fetch origin > /tmp/dgxff.log 2>&1; echo "FETCH-RC=$?"
git -C "$R" rev-parse origin/main
timeout 60 git -C "$R" checkout main > /tmp/dgxco.log 2>&1; echo "CHECKOUT-RC=$?"; tail -2 /tmp/dgxco.log
timeout 120 git -C "$R" merge --ff-only origin/main > /tmp/dgxmg.log 2>&1; echo "MERGE-RC=$?"; tail -3 /tmp/dgxmg.log
echo "--- HEAD-AFTER ---"
git -C "$R" rev-parse HEAD
echo "--- MD5-OLD-vs-NEW ---"
for f in $FILES; do
  O=$(git -C "$R" show "$OLD:$f" 2>/dev/null | md5sum | awk '{print $1}')
  N=$(md5sum "$R/$f" 2>/dev/null | awk '{print $1}')
  S=$(stat -c%s "$R/$f" 2>/dev/null)
  if [ -z "$O" ] || [ -z "$N" ]; then echo "MISSING $f old=$O new=$N bytes=$S"; continue; fi
  if [ "$O" = "$N" ]; then CH=UNCHANGED; else CH=CHANGED; fi
  echo "$CH $f old=$O new=$N bytes=$S"
done
echo "--- S26STAGE-CODE-BEFORE ---"
(cd "$ST/code" && md5sum $FILES 2>&1)
echo "--- PUSH-WORKER-STATUS ---"
ps -eo pid,etime,args | grep '[r]sync.*47.117.116.55' || echo NO-PUSH-RSYNC
tail -4 /home/arm/junyang/s30xfer.log 2>/dev/null || echo NO-XFER-LOG
echo "--- S26STAGE-SYNC ---"
if ps -eo args | grep -q '[r]sync.*47.117.116.55'; then
  echo SYNC-DEFERRED push-worker-still-running
else
  rsync -a "$R/code/" "$ST/code/" > /tmp/dgxrs.log 2>&1; echo "RSYNC-RC=$?"
  tail -3 /tmp/dgxrs.log
  echo "--- S26STAGE-CODE-AFTER ---"
  (cd "$ST/code" && md5sum $FILES 2>&1)
fi
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
