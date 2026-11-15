#!/bin/bash
# C2 line SCP to spot-31 (100.98.67.20) via egress proxy CONNECT. Usage: s31scp.sh <local> root@100.98.67.20:<remote>
exec scp -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new "$@"
