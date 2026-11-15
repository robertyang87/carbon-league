echo H24E2SYNC1942_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGX-BEGIN $(date '+%F %T %Z')"
cd /home/arm/junyang/carbon-repo || exit 1
git rev-parse --abbrev-ref HEAD; git rev-parse HEAD
md5sum code/eval_v53.py code/bigeval.py
M=$(md5sum code/eval_v53.py | cut -d' ' -f1)
if [ "$M" != "6d24da0eeeb28bad350d8b38f08ec3fc" ]; then
  echo "EVAL-MD5-MISMATCH $M -- trying pull"
  timeout 60 git pull --ff-only 2>&1 | tail -2
  md5sum code/eval_v53.py code/bigeval.py
  M=$(md5sum code/eval_v53.py | cut -d' ' -f1)
fi
if [ "$M" = "6d24da0eeeb28bad350d8b38f08ec3fc" ]; then
  timeout 120 scp -o BatchMode=yes -o ConnectTimeout=20 code/eval_v53.py root@47.117.16.90:/root/carbon/code/eval_v53.py && echo SCP-OK
  timeout -k 10 60 ssh -o BatchMode=yes -o ConnectTimeout=20 root@47.117.16.90 "md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py"
else
  echo "EVAL-MD5-STILL-MISMATCH $M -- NO SYNC"
fi
echo "DGX-END"
DGXEOF
echo "L20-END rc=$?"
echo H24E2SYNC1942_END
