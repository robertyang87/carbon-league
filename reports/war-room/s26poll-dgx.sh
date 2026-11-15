echo S26POLL_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'tail -5 /home/arm/junyang/s26xfer.log; echo ---; du -sh /home/arm/junyang/s26stage 2>/dev/null; pgrep -fc rsync || echo 0' < /dev/null 2>&1 | head -20
echo S26POLL_END
