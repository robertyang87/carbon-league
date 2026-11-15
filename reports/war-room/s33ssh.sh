#!/bin/bash
# C2 line driver: Tailscale SSH direct to spot-33 (100.90.180.19) via egress proxy CONNECT.
# Usage: s33ssh.sh 'remote command'   |   s33ssh.sh 'bash -s' < local_script.sh
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
  root@100.90.180.19 "$@"
