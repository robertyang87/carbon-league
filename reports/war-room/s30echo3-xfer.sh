#!/bin/bash
# Runs on L20 via cloud-assistant; DGX pushes echo3 pack to spot-30 and verifies.
MARK=S30E3XFER0247
echo "$MARK PAYLOAD-BEGIN"
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK=S30E3XFER0247
SRC=/home/arm/junyang/s26stage/echo3_int8.py
TGT=47.117.116.55
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
EXP_MD5=7b7620499411a088cf4dc17c0e929bd7
EXP_SIZE=4576790
echo "$MARK DGX-BEGIN $(date '+%F %T %Z')"
echo "--- SRC-VERIFY ---"
ls -l "$SRC" || { echo SRC-MISSING; echo "$MARK PAYLOAD-END"; exit 0; }
SM=$(md5sum "$SRC" | awk '{print $1}')
SS=$(stat -c %s "$SRC")
echo "SRC-MD5=$SM SRC-SIZE=$SS"
if [ "$SM" != "$EXP_MD5" ] || [ "$SS" != "$EXP_SIZE" ]; then echo SRC-MISMATCH; echo "$MARK PAYLOAD-END"; exit 0; fi
echo SRC-OK
echo "--- PACK-STRUCTURE ---"
grep -c "def agent" "$SRC" || true
grep -m1 -n "_OPTS" "$SRC" | cut -c1-300 || echo NO-OPTS-LINE
echo "--- MKDIR-ON-TARGET ---"
timeout -k 30 120 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'mkdir -p /root/carbon/data/packs && echo MKDIR-OK' || { echo TARGET-UNREACHABLE; echo "$MARK PAYLOAD-END"; exit 0; }
echo "--- SCP ---"
timeout -k 30 600 scp -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$SRC" root@"$TGT":/root/carbon/data/packs/echo3_int8.py
echo "SCP-RC=$?"
echo "--- TARGET-VERIFY ---"
timeout -k 30 120 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
F=/root/carbon/data/packs/echo3_int8.py
ls -l "$F"
echo "TGT-MD5=$(md5sum "$F" | awk '{print $1}') TGT-SIZE=$(stat -c %s "$F")"
ls -l /root/carbon/code/exp/opp_master2r.py 2>/dev/null || echo NO-OPP-MASTER2R
grep -c "def agent" "$F" || true
TGTEOF
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
echo "$MARK PAYLOAD-END"
