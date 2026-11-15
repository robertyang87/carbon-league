#!/bin/bash
# ppou2 control-arm mirror loop. Runs ON gpu-l20-6 (stable machine),
# pulls the control arm's live files from the spot machine every 120 s
# (owner 2026-11-15 14:13: 2-minute sync caliber, exam-field rule).
# A spot reclaim then loses at most ~2 minutes of state/log.
#
# Usage: sync_c_mirror.sh <spot_tailnet_ip>
# Landing: /root/carbon/data/rl3/ppou2_c_mirror/ on gpu-l20-6.
# Loop log: /root/carbon/data/rl3/c_mirror_sync.log (one line per cycle:
# timestamp, rsync rc, bytes received). Launch detached:
#   setsid nohup /root/carbon/sync_c_mirror.sh <ip> >/dev/null 2>&1 &
# Record the PGID in /root/carbon/data/rl3/c_mirror_sync.pgid at launch.
set -u
SPOT="${1:?spot tailnet ip}"
SRC_DIR=/root/carbon/data/rl3/ppou2_c
DST_DIR=/root/carbon/data/rl3/ppou2_c_mirror
LOG=/root/carbon/data/rl3/c_mirror_sync.log
SSH="ssh -i /root/.ssh/spot_sync -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
mkdir -p "$DST_DIR"
echo "$(date '+%F %T') LOOP_START spot=$SPOT dst=$DST_DIR" >> "$LOG"
while true; do
  T0=$(date +%s)
  # Live files: state, per-round lines, eval lines, league, the arm log,
  # any new checkpoints, and the probe file used for liveness checks.
  rsync -a --partial --timeout=90 -e "$SSH" \
    "root@${SPOT}:${SRC_DIR}/state_last.pt" \
    "root@${SPOT}:${SRC_DIR}/train.jsonl" \
    "root@${SPOT}:${SRC_DIR}/eval.jsonl" \
    "root@${SPOT}:${SRC_DIR}/league.json" \
    "root@${SPOT}:${SRC_DIR}/sync_probe.txt" \
    "$DST_DIR/" >> "$LOG" 2>&1
  RC1=$?
  rsync -a --partial --timeout=90 -e "$SSH" \
    "root@${SPOT}:/root/carbon/data/rl3/ppou2_c.log" \
    "$DST_DIR/ppou2_c.log" >> "$LOG" 2>&1
  RC2=$?
  rsync -a --partial --timeout=120 -e "$SSH" \
    --include='ckpt_*.npz' --include='state_anchor_frozen.pt' --exclude='*' \
    "root@${SPOT}:${SRC_DIR}/" "$DST_DIR/" >> "$LOG" 2>&1
  RC3=$?
  echo "$(date '+%F %T') cycle rc=$RC1/$RC2/$RC3 state_mtime=$(stat -c %y "$DST_DIR/state_last.pt" 2>/dev/null | cut -d. -f1) train_lines=$(wc -l < "$DST_DIR/train.jsonl" 2>/dev/null)" >> "$LOG"
  T1=$(date +%s)
  SLEEP=$((120 - (T1 - T0)))
  [ "$SLEEP" -gt 0 ] && sleep "$SLEEP"
done
