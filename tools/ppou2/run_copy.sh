#!/bin/bash
# Copy the ppou2 environment from gpu-l20-6 to the spot machine.
# Runs ON gpu-l20-6, detached; logs progress per item.
# The ppou2_c copy here is a warm pre-copy (arm may still be running);
# a final refresh after the arm stops makes it consistent.
set -u
DST=root@100.92.60.120
E="ssh -i /root/.ssh/spot_sync -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
LOG=/root/copy_to_spot.log
echo "$(date '+%F %T') COPY_START" >> "$LOG"
rsync -a --partial --timeout=120 -e "$E" /root/carbon/.venv "$DST:/root/carbon/" >> "$LOG" 2>&1
echo "$(date '+%F %T') venv rc=$?" >> "$LOG"
rsync -a --partial --timeout=120 -e "$E" /root/carbon/code "$DST:/root/carbon/" >> "$LOG" 2>&1
echo "$(date '+%F %T') code rc=$?" >> "$LOG"
rsync -a --partial --timeout=120 -e "$E" /root/carbon/data/weights "$DST:/root/carbon/data/" >> "$LOG" 2>&1
echo "$(date '+%F %T') weights rc=$?" >> "$LOG"
rsync -a --partial --timeout=120 -e "$E" /root/carbon/data/rl3/ppou2_c "$DST:/root/carbon/data/rl3/" >> "$LOG" 2>&1
echo "$(date '+%F %T') ppou2_c rc=$?" >> "$LOG"
rsync -a --partial --timeout=120 -e "$E" \
  /root/carbon/launch_ppou2_v5.py /root/carbon/remote_edit_kl.py \
  /root/carbon/remote_serial_check.py /root/carbon/remote_kl_check.py \
  /root/carbon/remote_sidecar_trees.py /root/carbon/FILING.md \
  "$DST:/root/carbon/" >> "$LOG" 2>&1
echo "$(date '+%F %T') loose rc=$? COPY_DONE" >> "$LOG"
