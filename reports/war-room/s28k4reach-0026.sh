echo S28K4REACH0026_BEGIN
date "+%F %T %Z"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGX-BEGIN"
curl -s -m 15 -o /dev/null -w "GITHUB_API_HTTP=%{http_code}\n" https://api.github.com || echo GITHUB_API_FAIL
ping -c 2 -W 3 47.116.17.244 || echo PING-S28-FAIL
timeout -k 10 45 ssh -i /home/arm/.ssh/id_ed25519_carbon_relay -o BatchMode=yes -o ConnectTimeout=30 root@47.116.17.244 'echo SPOT28-ALIVE; wc -l /root/carbon/data/bigeval_live/*.jsonl' || echo SSH-SPOT28-FAIL
echo "DGX-END"
DGXEOF
echo S28K4REACH0026_END
