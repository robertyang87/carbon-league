echo S26P326_BEGIN
date "+%F %T %Z"
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub 2>/dev/null | head -1
ls -la /root/carbon/
which rsync tar python3
echo S26P326_END
