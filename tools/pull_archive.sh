#!/bin/bash
set -x
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
rsync -a --partial --files-from=rundir_files_nolog.txt -e "$SSHOPT" root@100.79.251.87:/root/carbon/data/rl3/snipe_n1/ l20_archive_stage/run_dir/
echo "run_dir rsync rc=$?"
rsync -a --partial -e "$SSHOPT" root@100.79.251.87:/root/carbon/data/rl3/snipe_n1.log l20_archive_stage/run_dir/snipe_n1.log
echo "log rsync rc=$?"
rsync -a --partial --exclude=value-diag --exclude=__pycache__ -e "$SSHOPT" root@100.79.251.87:/root/ppo_attrib/ l20_archive_stage/ppo_attrib/
echo "attrib rsync rc=$?"
du -sb l20_archive_stage/run_dir l20_archive_stage/ppo_attrib
echo PULL_DONE
