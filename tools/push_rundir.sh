#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
rsync -a --partial -e "$SSHOPT" l20_archive_stage/run_dir/ arm@100.84.148.73:/home/arm/junyang/l20_ppo_u_archive_20261114/run_dir/
echo "push run_dir rc=$?"
echo PUSH_RUNDIR_DONE
