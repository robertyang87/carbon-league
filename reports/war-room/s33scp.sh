#!/bin/bash
# C2 line driver: scp to/from spot-33 (100.90.180.19) via egress proxy CONNECT.
# Usage: s33scp.sh <src...> <dst>  (same arg order as scp)
exec scp -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=60 -o StrictHostKeyChecking=accept-new \
  "$@"
