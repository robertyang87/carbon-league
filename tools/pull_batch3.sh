#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
rsync -a --partial --files-from=batch3_rundir_files.txt -e "$SSHOPT" root@100.79.251.87:/root/carbon/data/rl3/snipe_n1/ l20_archive_stage/run_dir/
echo "pull batch3 run_dir rc=$?"
mkdir -p l20_archive_stage/rl3
rsync -a --partial --files-from=batch3_rl3_files.txt -e "$SSHOPT" root@100.79.251.87:/root/carbon/data/rl3/ l20_archive_stage/rl3/
echo "pull batch3 rl3 rc=$?"
echo PULL3_DONE
