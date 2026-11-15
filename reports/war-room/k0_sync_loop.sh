#!/bin/bash
# C2 -> DGX shard return loop. Runs ON spot-31.
# Every ~120 s: push newly landed shards (stable >=60 s) + current progress.json
# to DGX /home/arm/junyang/k0_dist/returns/c2/. Stop file: /root/carbon/data/k0_sync_stop
OUT=/root/carbon/data/k0_annot_c2
DEST=arm@100.84.148.73:/home/arm/junyang/k0_dist/returns/c2/
KEY=/root/.ssh/id_ed25519
SENT=/root/carbon/data/k0_sync_sent.txt
LOG=/root/carbon/data/k0_sync.log
SSHOPT="-i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30"
touch "$SENT"
ssh $SSHOPT arm@100.84.148.73 'mkdir -p /home/arm/junyang/k0_dist/returns/c2' >>"$LOG" 2>&1
echo "$(date -Iseconds) SYNC_LOOP_START pid=$$" >> "$LOG"
while true; do
  now=$(date +%s)
  for f in "$OUT"/shards/*; do
    [ -f "$f" ] || continue
    b=$(basename "$f")
    grep -qx "$b" "$SENT" && continue
    mt=$(stat -c %Y "$f")
    [ $((now - mt)) -lt 60 ] && continue
    if scp $SSHOPT "$f" "$DEST" >>"$LOG" 2>&1; then
      echo "$b" >> "$SENT"
      echo "$(date -Iseconds) SENT $b" >> "$LOG"
    else
      echo "$(date -Iseconds) SEND_FAIL $b" >> "$LOG"
    fi
  done
  scp $SSHOPT "$OUT/progress.json" "${DEST}progress.json" >>"$LOG" 2>&1
  if [ -f /root/carbon/data/k0_sync_stop ]; then
    echo "$(date -Iseconds) STOP seen, exiting" >> "$LOG"
    break
  fi
  sleep 120
done
