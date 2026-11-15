#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
rsync -a --partial --exclude=__pycache__ -e "$SSHOPT" root@100.79.251.87:/root/ppo_attrib/value-diag/ l20_archive_stage/ppo_attrib/value-diag/
echo "pull batch2 rc=$?"
echo PULL2_DONE
