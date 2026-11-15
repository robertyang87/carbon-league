echo S26POLL8_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'grep -E "PUSH rc|S26XFER DONE" /home/arm/junyang/s26xfer.log; pgrep -fc rsync || echo 0rsync' < /dev/null 2>&1 | head -8
echo S26POLL8_END
