#!/bin/bash
# rl_sync.sh HOST OUT [REMOTE_ROOT] [MAX_LOCAL_GAMES]  -- Mac side of the split PPO loop, every 10 s:
#   pull finished games  cloud:REMOTE_ROOT/out/*  ->  OUT/remote_in/   (removed on the cloud after transfer)
#   push learner state   OUT/live/w_*.npz, OUT/league.json, OUT/snap_*.npz  ->  cloud:REMOTE_ROOT/in/
H=$1; OUT=$2; R=${3:-carbon/data/rlc}; MAXG=${4:-200}; K=~/.ssh/carbon_aliyun_ed25519
E="ssh -i $K -o ConnectTimeout=20 -o ServerAliveInterval=15"
mkdir -p "$OUT/remote_in"
ssh -i $K root@$H "mkdir -p $R/in/live $R/in/snaps $R/out"
while true; do
  rsync -a --timeout=120 -e "$E" "$OUT/live/" root@$H:$R/in/live/ --delete 2>>"$OUT/sync.err"
  [ -f "$OUT/league.json" ] && rsync -a --timeout=60 -e "$E" "$OUT/league.json" root@$H:$R/in/league.json 2>>"$OUT/sync.err"
  ls "$OUT"/snap_*.npz >/dev/null 2>&1 && rsync -a --timeout=120 -e "$E" "$OUT"/snap_*.npz root@$H:$R/in/snaps/ 2>>"$OUT/sync.err"
  have=$(ls "$OUT/remote_in" | sed -n 's/.*_\([0-9]*\)\.pkl\.z$/\1/p' | awk '{s+=$1} END {print s+0}')
  # only pull while the learner is short of games (cloud egress is billed per GB; the actor keeps the newest)
  [ "$have" -lt "$MAXG" ] && rsync -a --timeout=300 --remove-source-files -e "$E" root@$H:$R/out/ "$OUT/remote_in/" 2>>"$OUT/sync.err"
  sleep 10
done
