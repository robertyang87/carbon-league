echo S26POLL4_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'grep -E "rc=|STAGE|DONE" /home/arm/junyang/s26xfer.log | tail -8; echo ---; du -sh /home/arm/junyang/s26stage 2>/dev/null; pgrep -fc rsync || echo 0; timeout 30 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@8.133.222.114 "du -sh /root/carbon 2>/dev/null" 2>/dev/null || echo S26_NOT_YET' < /dev/null 2>&1 | head -20
echo S26POLL4_END
