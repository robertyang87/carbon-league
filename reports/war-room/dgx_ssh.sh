#!/bin/bash
# DGX direct-read helper (read-only). Usage: dgx_ssh.sh "<remote command>"
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts \
  -o ConnectTimeout=30 arm@100.84.148.73 "$@"
