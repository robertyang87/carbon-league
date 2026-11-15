#!/bin/bash
# Phase B3 (ride-along only): spot-29 Echo2 md5 recheck + smoke JSON read-back + deployed md5 re-verify.
# Marker: S30DEPLOY0245B
MARK=S30DEPLOY0245B
echo "$MARK L20-BEGIN $(date '+%Y-%m-%d %H:%M:%S %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30DEPLOY0245B
echo "$MARK DGX-BEGIN $(date '+%Y-%m-%d %H:%M:%S %Z')"
TGT=47.117.170.99
KEY=""
for K in "$HOME"/.ssh/id_*; do
  case "$K" in *.pub|*known_hosts*) continue;; esac
  [ -f "$K" ] || continue
  if timeout -k 30 60 ssh -n -i "$K" -o BatchMode=yes -o ConnectTimeout=15 root@"$TGT" 'echo SSH-OK $(hostname)' </dev/null 2>/dev/null; then KEY="$K"; break; fi
done
if [ -z "$KEY" ]; then echo "$MARK SSH-NOT-READY"; echo "$MARK DGX-END"; exit 0; fi
echo "KEY-USED=$KEY"
SSHCMD="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20"
echo "--- DEPLOYED-MD5-REVERIFY (spot-29) ---"
timeout -k 30 120 $SSHCMD root@"$TGT" 'cd /root/carbon/code && md5sum eval_v53.py bigeval.py cnet2_search.py cnet2_build.py cnet2_search_timing.py replica_chain.sh rl2_ppo_rollout.py cnet2_np.py' 2>&1
echo "--- SMOKE-SUMMARY-READBACK (s29deploy_smoke.out) ---"
timeout -k 30 60 $SSHCMD root@"$TGT" 'grep -a "\"tag\": \"s29deploy_smoke\"" /root/carbon/data/s29deploy_smoke.out | tail -1' 2>&1
echo "--- RIDE-ALONG Echo2 md5 (spot-29) ---"
timeout -k 30 120 $SSHCMD root@"$TGT" 'md5sum /root/carbon/code/exp/echo2.py /root/carbon/data/weights/cnet2_m3w5L.npz' 2>&1
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
