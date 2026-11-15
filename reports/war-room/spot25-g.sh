#!/bin/bash
# spot25 step G (read-only on h24): how does the 2-min sync work there?
MARK="SPOT25G9K1"
echo "MARKER $MARK START"
date '+%F %T %Z'
timeout 120 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
H24=47.117.16.90
ssh -n -i "$KEY" -o BatchMode=yes -o ConnectTimeout=15 root@"$H24" 'echo ---CRON---; crontab -l 2>/dev/null; echo ---ROOT-LS---; ls -la /root/ /root/carbon/ | head -40; echo ---SYNC-PS---; ps -eo pid,etime,args | grep -iE "sync|git" | grep -v grep | head; echo ---SSH-DIR---; ls -la /root/.ssh/; echo ---GIT-CFG---; git -C /root/carbon/repo config --list 2>/dev/null | head; ls -d /root/carbon/*/ 2>/dev/null'
DGXEOF
echo "DGX-SSH-RC=$?"
echo "MARKER $MARK END"
