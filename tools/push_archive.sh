#!/bin/bash
set -x
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
DEST=arm@100.84.148.73:/home/arm/junyang/l20_ppo_u_archive_20261114
rsync -a --partial -e "$SSHOPT" l20_archive_stage/run_dir/ $DEST/run_dir/
echo "push run_dir rc=$?"
rsync -a --partial -e "$SSHOPT" l20_archive_stage/ppo_attrib/ $DEST/ppo_attrib/
echo "push attrib rc=$?"
rsync -a --partial -e "$SSHOPT" batch1_manifest.txt $DEST/MANIFEST.md5
echo "push manifest rc=$?"
echo PUSH_DONE
