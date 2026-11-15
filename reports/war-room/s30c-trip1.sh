#!/bin/bash
# Phase C trip 1 (own implementation): deploy 8 harness files DGX -> spot-30,
# on-machine md5 readback vs manifest, then new-harness smoke (seed 999901, n=2).
# Runs on L20 via cloud-assistant; drives DGX by ssh (control only). File bytes
# move DGX -> spot-30 directly by rsync over DGX's own outbound ssh.
MARK=S30CT1A7F3
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 1200 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30CT1A7F3
ST=/home/arm/junyang/s26stage
R=/home/arm/junyang/carbon-repo
TGT=47.117.116.55
FILES="eval_v53.py bigeval.py cnet2_search.py cnet2_build.py cnet2_search_timing.py replica_chain.sh rl2_ppo_rollout.py cnet2_np.py"
declare -A WANT=(
  [eval_v53.py]=aa4c5bb63c879caffeb33b0b25c498d7
  [bigeval.py]=3547eb6abba5083c5aaa2645fcbf2a47
  [cnet2_search.py]=14230fa124e0a9b90d59d6df03c6765a
  [cnet2_build.py]=8cd8a3d8a9e3e59e8b3ac966e797382e
  [cnet2_search_timing.py]=c60390a36cc6e895eb9a725f89f5973c
  [replica_chain.sh]=07fa5e914a74ab07ab482b470facb911
  [rl2_ppo_rollout.py]=d49298c059ac080b1e14f327519cdec6
  [cnet2_np.py]=d95f63843511052bc8f7a82551ce31a1
)
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
SRC=""
for D in "$ST/code" "$R/code"; do
  ok=0
  echo "--- MD5 CHECK $D ---"
  for f in $FILES; do
    m=$(md5sum "$D/$f" 2>/dev/null | awk '{print $1}')
    if [ "$m" = "${WANT[$f]}" ]; then ok=$((ok+1)); echo "OK  $f $m"; else echo "BAD $f got=$m want=${WANT[$f]}"; fi
  done
  echo "MATCH-COUNT $D $ok/8"
  if [ "$ok" = "8" ] && [ -z "$SRC" ]; then SRC="$D"; fi
done
if [ -z "$SRC" ]; then echo "$MARK SOURCE-NOMATCH"; echo "$MARK DGX-END"; exit 1; fi
echo "SOURCE-USED=$SRC"
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new root@"$TGT" 'echo SSH-OK $(hostname)' 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK SSH-NOT-READY"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
SSHCMD="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
echo "--- TARGET-BEFORE-MD5 (informational) ---"
timeout -k 30 120 $SSHCMD root@"$TGT" "cd /root/carbon/code && md5sum $FILES" 2>&1
echo "--- INVENTORY exp/ + weights (spot-30) ---"
timeout -k 30 60 $SSHCMD root@"$TGT" 'ls /root/carbon/code/exp/ | grep -E "m3s2r1|master2|echo2" ; echo ---; ls /root/carbon/data/weights/ | grep -E "m3s2r1|m3w5" ; echo ---; ls /root/carbon/data/bigeval_live/ 2>/dev/null | head -5' 2>&1
echo "--- RSYNC 8 FILES ($SRC -> spot-30) ---"
(cd "$SRC" && timeout -k 30 300 rsync -a --timeout=120 -e "ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new" $FILES root@"$TGT":/root/carbon/code/ </dev/null)
echo "RSYNC-RC=$?"
echo "--- TARGET-AFTER-MD5 vs MANIFEST ---"
AFTER=$(timeout -k 30 120 $SSHCMD root@"$TGT" "cd /root/carbon/code && md5sum $FILES" 2>&1)
echo "$AFTER"
bad=0
for f in $FILES; do
  m=$(echo "$AFTER" | awk -v fn="$f" '$2==fn {print $1}')
  if [ "$m" != "${WANT[$f]}" ]; then echo "MISMATCH $f got=$m want=${WANT[$f]}"; bad=$((bad+1)); fi
done
if [ "$bad" != "0" ]; then echo "DEPLOY-MD5-MISMATCH bad=$bad"; echo "$MARK DGX-END"; exit 1; fi
echo "DEPLOY-MD5-ALL-MATCH"
echo "--- SMOKE: m3s2r1 vs Master2, 2 games, seed 999901 ---"
timeout -k 30 700 $SSHCMD root@"$TGT" 'cd /root/carbon/code && EVO_WORKERS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 ../.venv/bin/python bigeval.py exp/m3s2r1.py exp/opp_master2r.py 999901 2 s30deploy_smoke > /root/carbon/data/s30deploy_smoke.out 2>&1; echo SMOKE-RC=$?'
SUMLINE=$(timeout -k 30 60 $SSHCMD root@"$TGT" 'grep -a "\"tag\": \"s30deploy_smoke\"" /root/carbon/data/s30deploy_smoke.out | tail -1' 2>&1)
echo "SMOKE-SUMMARY-LINE: $SUMLINE"
echo "$SUMLINE" | python3 -c '
import sys, json
line = sys.stdin.read().strip()
try:
    d = json.loads(line[line.index("{"):])
    print("SMOKE-PARSED n=%s err=%s wins=%s mean=%s se=%s worst_ms=%s" % (d.get("n"), d.get("err"), d.get("wins"), d.get("mean"), d.get("se"), d.get("worst_ms")))
    print("SMOKE-OK" if (d.get("err") == 0 and d.get("n") == 2 and not d.get("FAIL")) else "SMOKE-FAIL")
except Exception as e:
    print("SMOKE-PARSE-ERROR", e)
    print("SMOKE-FAIL")
'
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
