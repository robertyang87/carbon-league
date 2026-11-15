echo S26P26_BEGIN
date "+%F %T %Z"; hostname; uptime
echo "--- disk ---"; df -h / | tail -2
echo "--- /root ---"; ls -la /root/ | head
echo "--- authorized_keys ---"; wc -l /root/.ssh/authorized_keys
echo S26P26_END
