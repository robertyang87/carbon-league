echo H24E2AK1952_BEGIN
date "+%F %T %Z"
echo "--- AUTHORIZED KEYS FINGERPRINTS ---"
ssh-keygen -lf /root/.ssh/authorized_keys 2>/dev/null || echo NO-AK-FILE
wc -l /root/.ssh/authorized_keys 2>/dev/null
echo H24E2AK1952_END
