#!/bin/bash
SSHOPT='ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=yes -o ServerAliveInterval=30 -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p"'
DEST=arm@100.84.148.73:/home/arm/junyang/l20_ppo_u_archive_20261114
for i in 1 2 3 4 5 6; do
  rsync -a --partial --files-from=batch3_rundir_files.txt -e "$SSHOPT" l20_archive_stage/run_dir/ $DEST/run_dir/
  rc=$?; echo "push run_dir attempt $i rc=$rc"; [ $rc -eq 0 ] && break; sleep 10
done
rsync -a --partial -e "$SSHOPT" l20_archive_stage/run_dir/live l20_archive_stage/run_dir/backup_np_pre_s1v11 $DEST/run_dir/
echo "push subdirs rc=$?"
for i in 1 2 3; do
  rsync -a --partial --files-from=batch3_rl3_files.txt -e "$SSHOPT" l20_archive_stage/rl3/ $DEST/rl3/
  rc=$?; echo "push rl3 attempt $i rc=$rc"; [ $rc -eq 0 ] && break; sleep 10
done
rsync -a -e "$SSHOPT" batch3_manifest.txt $DEST/MANIFEST_full_series.md5
echo "push manifest3 rc=$?"
echo PUSH3_DONE
