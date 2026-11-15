#!/bin/bash
# Runs on L20 via cloud assistant; relays to DGX over ssh. READ ONLY Echo2 hunt on DGX.
MARK="ECHO2PREPK7QDGX"
echo "MARKER $MARK START"
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
date '+%F %T %Z'; hostname
echo "--- FIND ECHO ---"
timeout 60 find /home/arm/junyang/carbon -iname '*echo*' 2>/dev/null
echo "--- EXP LIST ---"
ls /home/arm/junyang/carbon/code/exp/ 2>/dev/null
echo "--- WEIGHTS (echo grep) ---"
ls -la /home/arm/junyang/carbon/data/weights/ | grep -i echo
echo "--- PACKS LS ---"
ls -la /home/arm/junyang/carbon/data/packs/ 2>/dev/null
echo "--- ECHO MD5 ---"
timeout 60 find /home/arm/junyang/carbon -iname '*echo*' -type f -exec md5sum {} \; 2>/dev/null
echo "--- REPO TREE ECHO (carbon-repo) ---"
cd /home/arm/junyang/carbon-repo && timeout 60 git ls-tree -r --name-only HEAD | grep -i echo
echo "--- GREP ECHO2 IN LIVE REPORTS (repo) ---"
timeout 60 grep -rn -i 'echo2' /home/arm/junyang/carbon-repo/reports/live/ 2>/dev/null | head -40
DGXEOF
echo "SSH_RC=$?"
echo "MARKER $MARK END"
