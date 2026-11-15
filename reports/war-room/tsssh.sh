#!/bin/bash
# Generic Tailscale SSH: tsssh.sh <user> <100.x> <command...>
U="$1"; H="$2"; shift 2
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o BatchMode=yes -o ConnectTimeout=30 -o StrictHostKeyChecking=accept-new \
  "$U@$H" "$@"
