#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
for i in 1 2 3 4 5 6; do
  rsync -a --partial --files-from=batch3_rundir_files.txt -e "$SSHOPT" root@100.79.251.87:/root/carbon/data/rl3/snipe_n1/ l20_archive_stage/run_dir/
  rc=$?
  echo "attempt $i rc=$rc"
  [ $rc -eq 0 ] && break
  sleep 10
done
echo PULL3B_DONE
