echo S26POLL3_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'grep -E "rc=|STAGE|DONE" /home/arm/junyang/s26xfer.log | tail -8; tail -2 /home/arm/junyang/s26xfer.log; echo ---; du -sh /home/arm/junyang/s26stage /home/arm/junyang/s26stage/data/weights 2>/dev/null; pgrep -fc rsync || echo 0' < /dev/null 2>&1 | head -20
echo S26POLL3_END
