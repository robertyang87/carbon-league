#!/bin/bash
# Deploy new harness — Phase C: spot-30 (47.117.116.55).
# Source of truth: DGX clone at dde9f2f4. The 8 harness files are byte-identical
# between 639253c0 and dde9f2f4 (compare file list: only data + code/cloud +
# code/packs/build_v17.sh changed). Manifest md5s (Phase A):
# eval_v53 aa4c5bb63c879caffeb33b0b25c498d7 / bigeval 3547eb6abba5083c5aaa2645fcbf2a47
# cnet2_search 14230fa124e0a9b90d59d6df03c6765a / cnet2_build 8cd8a3d8a9e3e59e8b3ac966e797382e
# cnet2_search_timing c60390a36cc6e895eb9a725f89f5973c / replica_chain.sh 07fa5e914a74ab07ab482b470facb911
# rl2_ppo_rollout d49298c059ac080b1e14f327519cdec6 / cnet2_np d95f63843511052bc8f7a82551ce31a1
# Marker: S30DEPLOY0225C
MARK=S30DEPLOY0225C
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 1200 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30DEPLOY0225C
R=/home/arm/junyang/carbon-repo
ST=/home/arm/junyang/s26stage
TGT=47.117.116.55
FILES="eval_v53.py bigeval.py cnet2_search.py cnet2_build.py cnet2_search_timing.py replica_chain.sh rl2_ppo_rollout.py cnet2_np.py"
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo "--- CLONE-HEAD ---"
git -C "$R" rev-parse HEAD
echo "--- SOURCE-MD5 (clone) ---"
(cd "$R/code" && md5sum $FILES > /tmp/s30dep-src.md5)
cat /tmp/s30dep-src.md5
echo "--- S26STAGE-CODE-MD5 (corrected paths, Phase A record fix) ---"
(cd "$ST/code" && md5sum $FILES)
echo "--- OBSERVE n1r pack in clone (read-only) ---"
ls -l "$R/data/packs/n1r_int8.py" 2>&1
md5sum "$R/data/packs/n1r_int8.py" 2>&1
grep -a -m1 -A6 '_OPTS' "$R/data/packs/n1r_int8.py" 2>/dev/null | head -10 || echo NO-OPTS-FOUND
echo "--- TARGET-BEFORE-MD5 ---"
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'echo SSH-OK $(hostname)' 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK SSH-NOT-READY"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
SSHCMD="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20"
RSYNCSSH="ssh -i $KEY -o BatchMode=yes -o ConnectTimeout=20"
timeout 30 ssh-keyscan -t ed25519,rsa "$TGT" > /tmp/s30dep-hk.txt 2>/dev/null </dev/null
if [ -s /tmp/s30dep-hk.txt ]; then
  grep -v -F "$TGT" ~/.ssh/known_hosts > /tmp/kh2.txt 2>/dev/null || true
  cat /tmp/s30dep-hk.txt >> /tmp/kh2.txt
  mv /tmp/kh2.txt ~/.ssh/known_hosts; chmod 600 ~/.ssh/known_hosts
  echo HOSTKEY-RECORDED
else
  echo KEYSCAN-FAILED-OR-CACHED
fi
timeout -k 30 120 $SSHCMD root@"$TGT" "cd /root/carbon/code && md5sum $FILES" 2>&1; echo "TARGET-BEFORE-RC=$? (replica_chain.sh absent on spot-30 is expected: never deployed there)"
echo "--- RSYNC 8 FILES (clone -> spot-30) ---"
(cd "$R/code" && timeout -k 30 300 rsync -a --timeout=120 -e "$RSYNCSSH" $FILES root@"$TGT":/root/carbon/code/)
echo "RSYNC-RC=$?"
echo "--- TARGET-AFTER-MD5 + DIFF ---"
timeout -k 30 120 $SSHCMD root@"$TGT" "cd /root/carbon/code && md5sum $FILES > /tmp/s30dep-tgt.md5 && cat /tmp/s30dep-tgt.md5" 2>&1
timeout -k 30 60 scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT":/tmp/s30dep-tgt.md5 /tmp/s30dep-tgt.md5 < /dev/null 2>&1
if diff -q /tmp/s30dep-src.md5 /tmp/s30dep-tgt.md5 >/dev/null 2>&1; then
  echo DEPLOY-MD5-ALL-MATCH
else
  echo DEPLOY-MD5-MISMATCH
  diff /tmp/s30dep-src.md5 /tmp/s30dep-tgt.md5 || true
  echo "$MARK DGX-END"
  exit 1
fi
echo "--- SMOKE: m3s2r1 vs Master2, 2 games, seed 999901 ---"
timeout -k 30 700 $SSHCMD root@"$TGT" 'cd /root/carbon/code && EVO_WORKERS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 600 ../.venv/bin/python bigeval.py exp/m3s2r1.py exp/opp_master2r.py 999901 2 s30deploy_smoke > /root/carbon/data/s30deploy_smoke.out 2>&1; echo SMOKE-RC=$?' > /tmp/s30smoke-rc.txt 2>&1
cat /tmp/s30smoke-rc.txt
timeout -k 30 60 $SSHCMD root@"$TGT" 'OUT=/root/carbon/data/s30deploy_smoke.out; tail -6 "$OUT"; SUM=$(grep -a "\"tag\": \"s30deploy_smoke\"" "$OUT" | tail -1); ERR=$(echo "$SUM" | sed -n "s/.*\"err\": \([0-9][0-9]*\).*/\1/p"); N=$(echo "$SUM" | sed -n "s/.*\"n\": \([0-9][0-9]*\).*/\1/p"); echo "SMOKE-SUMMARY err=$ERR n=$N"; if [ "$ERR" = "0" ] && [ "$N" = "2" ] && ! echo "$SUM" | grep -q "\"FAIL\": true"; then echo SMOKE-OK; else echo SMOKE-FAIL; fi' > /tmp/s30smoke-verdict.txt 2>&1
cat /tmp/s30smoke-verdict.txt
if grep -q "SMOKE-RC=0" /tmp/s30smoke-rc.txt && grep -q "SMOKE-OK" /tmp/s30smoke-verdict.txt; then
  echo SMOKE-PASS
else
  echo SMOKE-FAIL-STOP
  echo "$MARK DGX-END"
  exit 1
fi
echo "--- RIDE-ALONG Echo2 md5 (spot-30) ---"
timeout -k 30 60 $SSHCMD root@"$TGT" 'md5sum /root/carbon/code/exp/echo2.py /root/carbon/data/weights/cnet2_m3w5L.npz; echo EXPECTED echo2.py 15b6d754e6e21de33870074d3983a16f; echo EXPECTED cnet2_m3w5L.npz 82fde46b2633950d3cbb29e8cacbfdc2' 2>&1
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
