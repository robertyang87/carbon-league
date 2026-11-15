echo S26POLL5_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'grep -E "rc=|DONE" /home/arm/junyang/s26xfer.log | tail -6; grep -A7 "STAGE MD5" /home/arm/junyang/s26xfer.log | tail -7; pgrep -fc rsync || echo 0rsync; timeout 30 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@8.133.222.114 "du -sh /root/carbon /root/carbon/code /root/carbon/data/weights 2>/dev/null" 2>/dev/null || echo S26_SSH_FAIL' < /dev/null 2>&1 | head -24
echo S26POLL5_END
