#!/bin/bash
# War-room probe 3: fix the stale-fetch block found by r1fable-1533.
# 1533 found: DGX repo HEAD stuck at a33d21c (~13:15); git fetch times out at 60s;
# orphaned `git fetch origin` PID 115896 on DGX (from timed-out probe 1, ~15:23);
# L20 leftovers PIDs 150212/150213 (vfy2 scp/sftp, ~14:46) + 152458 (probe-1 ssh).
# This probe: kill the stale fetch (PID-verified) + L20 leftovers, re-fetch with
# timeout 180, pull, copy Fable+Carago corpora, dump m3s2.sh/m7bs.sh/build_top.py.
# Marker: R1FABLE1545X7
MARK=R1FABLE1545X7
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout 420 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "R1FABLE1545X7 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- DGX-KILL-STALE-FETCH ---'
ps -p 115896 -o pid,etime,args= 2>/dev/null || echo FETCH115896-GONE
if ps -p 115896 -o args= 2>/dev/null | grep -q 'git fetch origin'; then kill 115896; sleep 2; ps -p 115896 -o pid,args= 2>/dev/null || echo FETCH115896-KILLED; fi
echo '--- DGX-GIT-STATE-BEFORE ---'
cd ~/junyang/carbon-repo && git rev-parse HEAD && git log --oneline -2
echo '--- REPO-PULL-BOUNDED-180 ---'
timeout 180 git fetch origin; echo "fetch_rc=$?"
git checkout cnet2-search-v12 2>&1 | tail -1
timeout 120 git pull --ff-only origin cnet2-search-v12 2>&1 | tail -3; echo "pull_rc=$?"
git log --oneline -3
if [ "$fetch_rc" = "124" ]; then echo '--- FETCH-STILL-HUNG-DIAG ---'; timeout 30 git ls-remote origin HEAD; echo "lsremote_rc=$?"; fi
echo '--- CORPUS-COPY ---'
mkdir -p /home/arm/junyang/carbon/data/fable_corpus/replays /home/arm/junyang/carbon/data/carago_corpus/replays
cp -n ~/junyang/carbon-repo/data/fable_corpus/*.json.gz /home/arm/junyang/carbon/data/fable_corpus/replays/ 2>&1 | head -3
cp -n ~/junyang/carbon-repo/data/fable_matches.json /home/arm/junyang/carbon/data/fable_matches.json 2>&1 | head -3
cp -n ~/junyang/carbon-repo/data/carago_corpus/*.json.gz /home/arm/junyang/carbon/data/carago_corpus/replays/ 2>&1 | head -3
cp -n ~/junyang/carbon-repo/data/carago_matches.json /home/arm/junyang/carbon/data/carago_matches.json 2>&1 | head -3
echo -n 'fable_replays_after='; ls /home/arm/junyang/carbon/data/fable_corpus/replays/ 2>/dev/null | wc -l
echo -n 'carago_replays_after='; ls /home/arm/junyang/carbon/data/carago_corpus/replays/ 2>/dev/null | wc -l
ls /home/arm/junyang/carbon/data/fable_matches.json /home/arm/junyang/carbon/data/carago_matches.json 2>&1
echo '--- GPU-IDLE-CHECK ---'
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SCRIPT-DUMP-M3S2 ---'
cat /home/arm/junyang/carbon/code/m3s2.sh
echo '--- SCRIPT-DUMP-M7BS ---'
cat /home/arm/junyang/carbon/code/m7bs.sh
echo '--- SCRIPT-DUMP-BUILD_TOP-HEAD ---'
sed -n '1,120p' /home/arm/junyang/carbon/code/build_top.py
echo '--- TEACHER-REGREP ---'
grep -n "TOP_TEACHERS\|TEACHERS" /home/arm/junyang/carbon/code/build_top.py /home/arm/junyang/carbon/code/cnet2_build.py /home/arm/junyang/carbon/code/cnet2_feat.py 2>/dev/null | head -20
echo "R1FABLE1545X7 DGX-END"
DGXEOF
echo "$MARK SSH_RC=$? L20-END"
echo '--- L20-KILL-LEFTOVERS ---'
for pid in 150212 150213 152458; do
  args=$(ps -p $pid -o args= 2>/dev/null)
  if [ -n "$args" ]; then echo "PID $pid: $args"; case "$args" in *vfy2*|*dgx_access*) kill $pid && echo "KILLED $pid";; *) echo "SKIP $pid (cmdline mismatch)";; esac; else echo "PID $pid gone"; fi
done
ps -eo pid,etime,args | grep -E 'ssh.*10.211.43.155' | grep -v grep || echo NO-STALE-L20
echo "$MARK L20-DONE"
