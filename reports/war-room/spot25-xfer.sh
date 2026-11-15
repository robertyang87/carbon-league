#!/bin/bash
# Runs ON DGX (detached). Transfer exam stack h24 -> DGX -> spot-25, verify.
set -u
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H24=47.117.16.90
S25=47.116.17.244
W=/home/arm/s25tmp
mkdir -p "$W/w"
SSH="ssh -n -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
SCP="scp -i $KEY -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
step(){ echo "[$(date '+%T')] $*"; }
step "START df:"; df -h /home/arm | tail -1
step "FIX-KNOWN-HOSTS"
ssh-keygen -f /home/arm/.ssh/known_hosts -R "$S25" >/dev/null 2>&1
KS=$(ssh-keyscan -t ed25519 "$S25" 2>/dev/null)
echo "$KS" >> /home/arm/.ssh/known_hosts
FP=$(echo "$KS" | ssh-keygen -lf /dev/stdin 2>/dev/null | awk '{print $2}')
echo "S25-FP $FP"
if [ "$FP" != "SHA256:d/TRPRl0YaBAX+up+U0ErUIm96aO0EpWebNL4CqZxwY" ]; then echo "XFER-FAIL hostkey-mismatch $FP"; exit 1; fi
step "S25-SSH-TEST"
$SSH root@"$S25" 'hostname' || { echo "XFER-FAIL s25-ssh"; exit 1; }
step "TAR-ON-H24"
$SSH root@"$H24" 'rm -f /tmp/s25stack.tar; tar -C /root -cf /tmp/s25stack.tar --exclude=carbon/code/m2fit --exclude=carbon/code/mfit --exclude=carbon/code/__pycache__ --exclude=carbon/code/root@\* carbon/code carbon/.venv && ls -la /tmp/s25stack.tar' || { echo "XFER-FAIL tar"; exit 1; }
step "PULL-TAR-TO-DGX"
$SCP root@"$H24":/tmp/s25stack.tar "$W/s25stack.tar" </dev/null || { echo "XFER-FAIL pull-tar"; exit 1; }
ls -la "$W/s25stack.tar"
step "PUSH-TAR-TO-S25"
$SCP "$W/s25stack.tar" root@"$S25":/root/s25stack.tar </dev/null || { echo "XFER-FAIL push-tar"; exit 1; }
step "PULL-WEIGHTS-TO-DGX"
for f in cnet2_m3w5L.npz cnet2_m3w3L.npz cnet2_m3s2r1L.npz; do
  $SCP root@"$H24":/root/carbon/data/weights/$f "$W/w/$f" </dev/null || { echo "XFER-FAIL w-pull $f"; exit 1; }
done
md5sum "$W"/w/*
step "PUSH-WEIGHTS-AND-UNTAR-ON-S25"
$SSH root@"$S25" 'mkdir -p /root/carbon/data/weights /root/carbon/data/bigeval_live' || { echo "XFER-FAIL mkdir"; exit 1; }
$SCP "$W/w/cnet2_m3w5L.npz" "$W/w/cnet2_m3w3L.npz" "$W/w/cnet2_m3s2r1L.npz" root@"$S25":/root/carbon/data/weights/ </dev/null || { echo "XFER-FAIL w-push"; exit 1; }
$SSH root@"$S25" 'tar -C /root -xf /root/s25stack.tar && echo UNTAR-OK && du -sh /root/carbon/code /root/carbon/.venv' || { echo "XFER-FAIL untar"; exit 1; }
step "VERIFY-ON-S25"
$SSH root@"$S25" 'md5sum /root/carbon/data/weights/cnet2_m3w5L.npz /root/carbon/data/weights/cnet2_m3w3L.npz /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/code/exp/echo2.py /root/carbon/code/exp/echo_e0.py /root/carbon/code/exp/m3s2r1.py; /root/carbon/.venv/bin/python -c "import numpy; print(\"NUMPY\", numpy.__version__)"; cd /root/carbon/code && /root/carbon/.venv/bin/python -c "import cnet2_search; print(\"IMPORT-OK\")"' || { echo "XFER-FAIL verify"; exit 1; }
echo "XFER-DONE"
