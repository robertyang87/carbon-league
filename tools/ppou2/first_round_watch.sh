#!/bin/bash
# Poll L20 until both ppou2 arms have >=1 train.jsonl line; print those lines (first-round achieved KL).
SSH="ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p root@100.79.251.87"
for i in $(seq 1 200); do
  R=$(timeout -k 10 40 $SSH 'bash -c "
for a in m c; do
  d=/root/carbon/data/rl3/ppou2_\$a
  echo \$a train=\$(wc -l < \$d/train.jsonl) warm=\$(wc -l < \$d/warmup.jsonl) state=\$(test -f \$d/state_last.pt && echo Y || echo N) alive=\$(kill -0 91223 2>/dev/null && echo m_ok; kill -0 91224 2>/dev/null && echo c_ok)
done"' 2>/dev/null)
  echo "poll $i $(date +%T): $R"
  TM=$(echo "$R" | grep -o 'm train=[0-9]*' | grep -o '[0-9]*$')
  TC=$(echo "$R" | grep -o 'c train=[0-9]*' | grep -o '[0-9]*$')
  if [ "${TM:-0}" -ge 1 ] && [ "${TC:-0}" -ge 1 ]; then
    timeout -k 10 40 $SSH 'bash -c "head -1 /root/carbon/data/rl3/ppou2_m/train.jsonl; head -1 /root/carbon/data/rl3/ppou2_c/train.jsonl; tail -1 /root/carbon/data/rl3/ppou2_m/warmup.jsonl; tail -1 /root/carbon/data/rl3/ppou2_c/warmup.jsonl"'
    echo "FIRST_ROUND_BOTH_ARMS"
    break
  fi
  sleep 60
done
