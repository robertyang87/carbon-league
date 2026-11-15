echo S28K4REACH0020_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGX-BEGIN"
date "+%F %T %Z"
ping -c 2 -W 3 47.116.17.244 || echo PING-FAIL
timeout -k 10 60 ssh -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=30 root@47.116.17.244 'echo SPOT28-ALIVE; date "+%F %T %Z"; wc -l /root/carbon/data/bigeval_live/*.jsonl; ls -l /root/carbon/data/bigeval_live/' || echo SSH-SPOT28-FAIL
echo "DGX-END"
DGXEOF
echo S28K4REACH0020_END
