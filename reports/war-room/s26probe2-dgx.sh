echo S26P2DGX_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'ls ~/.ssh/; echo ---; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 root@47.117.16.90 hostname 2>&1 | tail -2; echo ---H24ABOVE---; timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=15 root@8.133.222.114 hostname 2>&1 | tail -3; echo ---S26ABOVE---' 2>&1 | head -30
echo S26P2DGX_END
