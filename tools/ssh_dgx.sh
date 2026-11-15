#!/bin/sh
exec ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o "ProxyCommand=nc -X connect -x hatch-egress-proxy:3130 %h %p" -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null "$@"
