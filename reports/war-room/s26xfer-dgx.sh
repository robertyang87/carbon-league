#!/bin/bash
exec >> /home/arm/junyang/s26xfer.log 2>&1
echo "=== S26XFER START $(date '+%F %T %Z') ==="
set -x
ST=/home/arm/junyang/s26stage
rm -rf $ST; mkdir -p $ST/data
SSHOPT="-i $HOME/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=20"
ssh-keygen -R 8.133.222.114 >/dev/null 2>&1
timeout 30 ssh-keyscan -t ed25519 8.133.222.114 >> $HOME/.ssh/known_hosts 2>/dev/null
rsync -a -e "ssh $SSHOPT" --exclude 'm2fit/' --exclude 'mfit/' --exclude 'root@*' root@47.117.16.90:/root/carbon/code/ $ST/code/
echo "PULL_CODE rc=$?"
rsync -a -e "ssh $SSHOPT" root@47.117.16.90:/root/carbon/.venv/ $ST/.venv/
echo "PULL_VENV rc=$?"
rsync -a -e "ssh $SSHOPT" root@47.117.16.90:/root/carbon/data/weights/ $ST/data/weights/
echo "PULL_WEIGHTS rc=$?"
cp /home/arm/junyang/carbon-repo/code/eval_v53.py $ST/code/eval_v53.py
echo "--- STAGE MD5 ---"
md5sum $ST/code/exp/echo2.py $ST/code/eval_v53.py $ST/code/bigeval.py $ST/data/weights/cnet2_m3w5L.npz $ST/data/weights/cnet2_m3w3L.npz $ST/data/weights/cnet2_s64.npz
du -sh $ST $ST/code $ST/.venv $ST/data/weights
rsync -a -e "ssh $SSHOPT" $ST/ root@8.133.222.114:/root/carbon/
echo "PUSH rc=$?"
echo "=== S26XFER DONE $(date '+%F %T %Z') ==="
