#!/bin/bash
# Self-play generation probe -- Mac Studio (re-issued 2026-11-13, plain text).
# Runs ON the Studio (driven over tailnet SSH by the Studio execution agent;
# the old L20-hop wrapper and its ZeroTier address are retired).
# Marker: SPROBE-STUDIO1. Window: 20 games or 180s, whichever first.
# Player: cnet2_search.make_search_agent + 13-key pack CONF (SEARCH off),
# weights cnet2_m3s2r1L.npz (md5 b6db08c9b076257c54817377bb93130d).
#
# Why this re-issue exists: the previous probe script carried selfplay_gen.py
# as an embedded base64 blob (old file sp-probe-studio.sh, md5
# cc84bdc243214a017cfaddde511b31c3), which could not be read back for review
# and is VOID -- do not run it. This version carries NO encoded blocks: the
# producer is a separate plain-text file in the same repo directory
# (code/producer/selfplay_gen.py, md5 b275e952da14d644e6fab41e4ce941c8),
# asserted below before running.
#
# Run contract (Studio execution agent):
#   1. Fetch BOTH files from carbon-league main:
#        code/producer/sp_probe_studio.sh   (this file)
#        code/producer/selfplay_gen.py
#   2. Put selfplay_gen.py at $HOME/junyang/carbon/code/selfplay_gen.py
#      (or leave it next to this script; both places are checked and the
#      file is md5-asserted either way).
#   3. Run on the Studio: bash sp_probe_studio.sh
# Writes only to $HOME/junyang/carbon/data/selfplay_probe/ .
# Studio is macOS: no GNU timeout -- the caller wraps the ssh call.
MARK=SPROBE-STUDIO1
set -u
VENV=$HOME/junyang/carbon/.venv/bin/python
DGX=arm@10.188.0.23
die() { echo "$MARK FAIL $1"; echo "$MARK STUDIO-END"; exit 1; }
echo "$MARK STUDIO-BEGIN $(date '+%F %T %Z') $(hostname)"
if [ ! -x "$VENV" ]; then die no-venv-python; fi
echo "VENV $($VENV -c 'import sys; print(sys.version.split()[0])')"
echo "CPU: $(sysctl -n hw.ncpu) cores; MEM: $(( $(sysctl -n hw.memsize) / 1073741824 ))Gi; LOAD: $(uptime | sed 's/.*load averages: //')"
CODE=$HOME/junyang/carbon/code
if [ ! -f "$CODE/cnet2_search.py" ] || [ ! -f "$CODE/fastsim.py" ] || [ ! -f "$CODE/eval_v53.py" ] || [ ! -f "$CODE/cnet2_feat.py" ] || [ ! -f "$CODE/cnet2_np.py" ] || [ ! -f "$CODE/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json" ]; then
  echo "CODE-INCOMPLETE, rsyncing code from DGX over LAN (rsync -e ssh without -n: the ssh stdin IS the rsync pipe)"
  mkdir -p "$CODE"
  rsync -avz --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20" $DGX:/home/arm/junyang/carbon-repo/code/ "$CODE/" || die rsync-code
fi
for f in cnet2_search.py fastsim.py eval_v53.py cnet2_feat.py cnet2_np.py; do
  echo "CODEMD5 $f $(md5 -q "$CODE/$f")"
done
WTS=$HOME/junyang/carbon/data/weights/cnet2_m3s2r1L.npz
if [ ! -f "$WTS" ]; then
  echo "WEIGHTS-MISSING locally, fetching from DGX over LAN (DGX-side filename is cnet2_m3s2r1_last.npz, same bytes)"
  mkdir -p $HOME/junyang/carbon/data/weights
  scp -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new $DGX:/home/arm/junyang/carbon/data/weights/cnet2_m3s2r1_last.npz "$WTS" || die scp-weights
fi
echo "WEIGHTS $WTS $(md5 -q "$WTS")"
if [ "$(md5 -q "$WTS")" != "b6db08c9b076257c54817377bb93130d" ]; then die weights-md5-mismatch; fi
# --- producer: plain-text file, md5-asserted. No encoded blocks anywhere. ---
GEN_MD5=b275e952da14d644e6fab41e4ce941c8
GEN_DST="$CODE/selfplay_gen.py"
if [ -f "$GEN_DST" ]; then
  echo "PRODUCER present $GEN_DST"
elif [ -f "$(dirname "$0")/selfplay_gen.py" ]; then
  cp "$(dirname "$0")/selfplay_gen.py" "$GEN_DST" || die producer-copy
  echo "PRODUCER copied from script directory to $GEN_DST"
else
  die producer-missing
fi
if [ "$(md5 -q "$GEN_DST")" != "$GEN_MD5" ]; then die script-md5-mismatch; fi
echo "SCRIPT-OK $GEN_DST md5=$GEN_MD5"
WORKERS=$(( $(sysctl -n hw.ncpu) - 2 ))
echo "WORKERS=$WORKERS LOAD-BEFORE: $(uptime | sed 's/.*load averages: //')"
"$VENV" "$GEN_DST" --weights "$WTS" --seed0 991000 --seed1 991200 --out $HOME/junyang/carbon/data/selfplay_probe/studio --workers "$WORKERS" --max-games 20 --max-seconds 180 --name m3s2r1 > /tmp/sprobe_studio.log 2>&1
RC2=$?
cat /tmp/sprobe_studio.log
echo "LOAD-AFTER: $(uptime | sed 's/.*load averages: //')"
grep -E "SPGEN-RESULT" /tmp/sprobe_studio.log || die no-SPGEN-RESULT
echo "$MARK STUDIO-END rc=$RC2"
exit $RC2
