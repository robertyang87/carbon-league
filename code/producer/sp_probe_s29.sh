#!/bin/bash
# Self-play generation probe -- spot-29 (re-issued 2026-11-13, plain text).
# Marker: SPROBE-S29-1. Window: 20 games or 180s, whichever first.
# Player: cnet2_search.make_search_agent + 13-key pack CONF (SEARCH off),
# weights cnet2_m3s2r1L.npz (md5 b6db08c9b076257c54817377bb93130d).
#
# Why this re-issue exists: the previous probe script carried selfplay_gen.py
# as an embedded base64 blob (old file sp-probe-s29.sh, md5
# 47e988702cce7a95b6181d8774ca526c), which could not be read back for review
# on the C1 line and is VOID -- do not run it. This version carries NO
# encoded blocks: the producer is a separate plain-text file in the same
# repo directory (code/producer/selfplay_gen.py, md5
# 95d2e29fbec8cfcebee57bf9bf0d25ed), asserted below before running.
#
# Run contract (C1):
#   1. Fetch BOTH files from carbon-league main:
#        code/producer/sp_probe_s29.sh   (this file)
#        code/producer/selfplay_gen.py
#   2. Put selfplay_gen.py at /root/carbon/code/selfplay_gen.py
#      (or leave it next to this script; both places are checked and the
#      file is md5-asserted either way).
#   3. Run: bash sp_probe_s29.sh
# Writes only to /root/carbon/data/selfplay_probe/ ; never touches sync/h29.
MARK=SPROBE-S29-1
set -u
echo "$MARK BEGIN $(date '+%F %T %Z') $(hostname)"
echo "CPU: $(nproc) cores; MEM: $(free -g | awk '/^Mem:/{print $2"Gi"}'); PY: $(/root/carbon/.venv/bin/python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo NO-VENV)"
CODE=/root/carbon/code
for f in cnet2_search.py fastsim.py eval_v53.py cnet2_feat.py cnet2_np.py; do
  if [ -f "$CODE/$f" ]; then echo "CODEMD5 $f $(md5sum "$CODE/$f" | awk '{print $1}')"; else echo "$MARK FAIL code-missing $f"; echo "$MARK END"; exit 1; fi
done
WTS=""
for c in /root/carbon/data/weights/cnet2_m3s2r1L.npz /root/carbon/code/../data/weights/cnet2_m3s2r1L.npz; do
  if [ -f "$c" ]; then WTS="$c"; break; fi
done
if [ -z "$WTS" ]; then echo "$MARK FAIL weights-missing cnet2_m3s2r1L.npz (searched /root/carbon/data/weights + code/../data/weights)"; echo "$MARK END"; exit 1; fi
echo "WEIGHTS $WTS $(md5sum "$WTS" | awk '{print $1}')"
if [ "$(md5sum "$WTS" | awk '{print $1}')" != "b6db08c9b076257c54817377bb93130d" ]; then echo "$MARK FAIL weights-md5-mismatch"; echo "$MARK END"; exit 1; fi
if [ ! -d "$CODE/env_src/carbon_challenge-master/zerosum_env/envs/carbon" ]; then echo "$MARK FAIL env_src-missing (eval_v53 needs carbon.json)"; echo "$MARK END"; exit 1; fi
# --- producer: plain-text file, md5-asserted. No encoded blocks anywhere. ---
GEN_MD5=95d2e29fbec8cfcebee57bf9bf0d25ed
GEN_DST="$CODE/selfplay_gen.py"
if [ -f "$GEN_DST" ]; then
  echo "PRODUCER present $GEN_DST"
elif [ -f "$(dirname "$0")/selfplay_gen.py" ]; then
  cp "$(dirname "$0")/selfplay_gen.py" "$GEN_DST" || { echo "$MARK FAIL producer-copy"; echo "$MARK END"; exit 1; }
  echo "PRODUCER copied from script directory to $GEN_DST"
else
  echo "$MARK FAIL producer-missing (fetch code/producer/selfplay_gen.py from carbon-league main)"; echo "$MARK END"; exit 1
fi
if [ "$(md5sum "$GEN_DST" | awk '{print $1}')" != "$GEN_MD5" ]; then echo "$MARK FAIL script-md5-mismatch"; echo "$MARK END"; exit 1; fi
echo "SCRIPT-OK $GEN_DST md5=$GEN_MD5"
WORKERS=$(( $(nproc) - 2 ))
[ "$WORKERS" -gt 56 ] && WORKERS=56
echo "WORKERS=$WORKERS LOAD-BEFORE: $(cat /proc/loadavg)"
CPU0=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8}' /proc/stat); IDLE0=$(awk '/^cpu /{print $5}' /proc/stat)
timeout -k 30 420 /root/carbon/.venv/bin/python "$GEN_DST" --weights "$WTS" --seed0 991200 --seed1 991400 --out /root/carbon/data/selfplay_probe/s29 --workers "$WORKERS" --max-games 20 --max-seconds 180 --name m3s2r1 2>&1 | tee /tmp/sprobe_s29.log
RC=${PIPESTATUS[0]}
CPU1=$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8}' /proc/stat); IDLE1=$(awk '/^cpu /{print $5}' /proc/stat)
echo "CPU-BUSY-PCT: $(( ( (CPU1-CPU0) - (IDLE1-IDLE0) ) * 100 / (CPU1-CPU0) ))% LOAD-AFTER: $(cat /proc/loadavg)"
grep -E "SPGEN-RESULT" /tmp/sprobe_s29.log || echo "$MARK FAIL no-SPGEN-RESULT rc=$RC"
echo "$MARK END rc=$RC"
exit $RC
