echo S26POLL7_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'grep -E "PUSH rc|S26XFER DONE" /home/arm/junyang/s26xfer.log | tail -3; pgrep -fc rsync || echo 0rsync; timeout 30 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@8.133.222.114 "du -sh /root/carbon 2>/dev/null" 2>/dev/null || echo S26_SSH_FAIL' < /dev/null 2>&1 | head -12
echo S26POLL7_END
