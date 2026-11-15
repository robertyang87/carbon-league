#!/bin/bash
# C2 line driver: Tailscale SSH direct to spot-30 (100.99.41.71) via egress proxy CONNECT.
# Usage: s30ssh.sh 'remote command'   |   s30ssh.sh 'bash -s' < local_script.sh
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
  root@100.99.41.71 "$@"
