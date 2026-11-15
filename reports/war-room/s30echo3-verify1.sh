#!/bin/bash
MARK=S30E3VERIFY0250
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
F=/root/carbon/data/packs/echo3_int8.py
if [ -f "$F" ]; then
  ls -l "$F"
  echo "TGT-MD5=$(md5sum "$F" | awk '{print $1}') TGT-SIZE=$(stat -c %s "$F")"
else
  echo PACK-NOT-ON-TARGET
fi
ls -l /root/carbon/code/exp/opp_master2r.py 2>/dev/null || echo NO-OPP-MASTER2R
echo "$MARK PAYLOAD-END"
