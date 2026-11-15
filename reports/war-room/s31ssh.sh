#!/bin/bash
# C2 line driver: Tailscale SSH direct to spot-31 (100.98.67.20) via egress proxy CONNECT.
# Usage: s31ssh.sh 'remote command'   |   s31ssh.sh 'bash -s' < local_script.sh
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
  root@100.98.67.20 "$@"
