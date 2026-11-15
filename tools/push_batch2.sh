#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
rsync -a --partial -e "$SSHOPT" l20_archive_stage/ppo_attrib/value-diag/ arm@100.84.148.73:/home/arm/junyang/l20_ppo_u_archive_20261114/ppo_attrib/value-diag/
echo "push batch2 rc=$?"
rsync -a -e "$SSHOPT" batch2_manifest.txt arm@100.84.148.73:/home/arm/junyang/l20_ppo_u_archive_20261114/MANIFEST_value_diag.md5
echo "push manifest2 rc=$?"
echo PUSH2_DONE
