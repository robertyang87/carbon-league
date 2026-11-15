#!/bin/bash
# spot25 step A: liveness + authorized_keys check.
MARK="SPOT25A7K1"
echo "MARKER $MARK START"
date '+%F %T %Z'
hostname; uptime
echo "--- CPU/MEM/DISK ---"
nproc; free -g | head -3; df -h / | tail -1
echo "--- AUTH KEYS ---"
wc -l < /root/.ssh/authorized_keys
awk '{print NR": "$NF}' /root/.ssh/authorized_keys
ssh-keygen -lf <(sed -n '2p' /root/.ssh/authorized_keys) 2>/dev/null || echo "LINE2-FP-FAIL"
echo "--- CLOUD ASSISTANT ---"
acs-agent --version 2>/dev/null || /usr/local/share/aliyun-assist/*/aliyun-service --version 2>/dev/null || echo "ACS-VERSION-UNKNOWN"
echo "--- PYTHON ---"
python3 --version 2>/dev/null; which python3
echo "MARKER $MARK END"
