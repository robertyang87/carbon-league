#!/bin/bash
# Acceptance #4: kill mid-run, resume, prove the finished shard is NOT
# recomputed (sha256 + mtime unchanged) and the generation completes.
set -u
cd ~/workspace/k0-round2/annotate/run
OUT=~/workspace/k0-round2/annotate/fixture/out2
rm -rf "$OUT"
python3 -c "import json; json.dump(['selfplay-994000','selfplay-994001','selfplay-994002'], open('/tmp/games3.json','w'))"
ARGS="--weights /home/hatch/workspace/s12work/heads_v2.npz --replays /home/hatch/workspace/gen1r_smoke/replays --games /tmp/games3.json --out $OUT --window-lo 206 --window-hi 208 --K 4 --M 2 --H 1 --workers 1"

echo "=== phase 1: run, kill after first shard lands"
python3 k0_annotate.py $ARGS > ~/workspace/k0-round2/annotate/fixture/resume_phase1.log 2>&1 &
PID=$!
for i in $(seq 1 120); do
  if [ -f "$OUT/shards/selfplay-994000.npz" ]; then sleep 3; break; fi
  sleep 5
done
ls -la "$OUT/shards/" || true
SHA1=$(sha256sum "$OUT/shards/selfplay-994000.npz" | cut -d' ' -f1)
MT1=$(stat -c %Y "$OUT/shards/selfplay-994000.npz")
echo "shard1 sha256=$SHA1 mtime=$MT1"
kill -9 $PID 2>/dev/null
wait $PID 2>/dev/null
echo "killed phase1 (exit=$?)"
cat "$OUT/progress.json" || true

echo "=== phase 2: resume"
python3 k0_annotate.py $ARGS > ~/workspace/k0-round2/annotate/fixture/resume_phase2.log 2>&1
echo "phase2 exit=$?"
cat ~/workspace/k0-round2/annotate/fixture/resume_phase2.log
SHA2=$(sha256sum "$OUT/shards/selfplay-994000.npz" | cut -d' ' -f1)
MT2=$(stat -c %Y "$OUT/shards/selfplay-994000.npz")
echo "shard1 after resume: sha256=$SHA2 mtime=$MT2"
if [ "$SHA1" = "$SHA2" ] && [ "$MT1" = "$MT2" ]; then
  echo "RESUME PROOF: shard1 byte-identical and untouched (not recomputed)"
else
  echo "RESUME PROOF FAILED: shard1 changed"
  exit 1
fi
grep -q "shard_skip_verified.*selfplay-994000" ~/workspace/k0-round2/annotate/fixture/resume_phase2.log \
  && echo "phase2 log confirms verified skip of shard1"
python3 k0_annotate.py --out "$OUT" --verify
