echo S26P3DGX_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 arm@10.211.43.155 'which rsync; timeout 30 ssh -i ~/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=15 root@47.117.16.90 hostname 2>&1 | tail -1; echo ---H24RELAY---; timeout 30 ssh-keyscan -t ed25519 8.133.222.114 2>/dev/null | ssh-keygen -lf - 2>/dev/null; echo ---SCAN26---; timeout 60 md5sum /home/arm/junyang/carbon-repo/code/eval_v53.py 2>/dev/null; cd /home/arm/junyang/carbon-repo && timeout 30 git log --oneline -1' 2>&1 | head -20
echo S26P3DGX_END
