#!/bin/bash
# War-room probe 2: bounded retry of the timed-out r1fable probe (run 43174815464,
# TIMEOUT after ~9.5 min, no output captured). r1 md5 gate already closed by the
# DGX 15:30 round, so this run only does: stale-process sweep, bounded git pull,
# Fable+Carago corpus copy into data/, build-tooling recon.
# Marker: R1FABLE1533K2 (claim only output containing this marker)
MARK=R1FABLE1533K2
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
echo '--- L20-STALE-SWEEP ---'
ps -eo pid,etime,args | grep -E 'ssh.*10.211.43.155|bash /tmp/r1fable|git.*fetch' | grep -v grep || echo NO-STALE-L20
timeout 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "R1FABLE1533K2 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- DGX-STALE-SWEEP ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-STALE-GIT
echo '--- GPU-IDLE-CHECK ---'
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- CORPUS-ON-DGX-BEFORE ---'
echo -n 'fable_replays='; ls /home/arm/junyang/carbon/data/fable_corpus/replays/ 2>/dev/null | wc -l
echo -n 'carago_replays='; ls /home/arm/junyang/carbon/data/carago_corpus/replays/ 2>/dev/null | wc -l
echo '--- REPO-PULL-BOUNDED ---'
cd ~/junyang/carbon-repo && git rev-parse --abbrev-ref HEAD
timeout 60 git fetch origin; echo "fetch_rc=$?"
git checkout cnet2-search-v12 2>&1 | tail -1
timeout 60 git pull --ff-only origin cnet2-search-v12 2>&1 | tail -3; echo "pull_rc=$?"
git log --oneline -2
echo '--- CORPUS-COPY ---'
mkdir -p /home/arm/junyang/carbon/data/fable_corpus/replays /home/arm/junyang/carbon/data/carago_corpus/replays
cp -n ~/junyang/carbon-repo/data/fable_corpus/*.json.gz /home/arm/junyang/carbon/data/fable_corpus/replays/
cp -n ~/junyang/carbon-repo/data/fable_matches.json /home/arm/junyang/carbon/data/fable_matches.json
cp -n ~/junyang/carbon-repo/data/carago_corpus/*.json.gz /home/arm/junyang/carbon/data/carago_corpus/replays/
cp -n ~/junyang/carbon-repo/data/carago_matches.json /home/arm/junyang/carbon/data/carago_matches.json
echo -n 'fable_replays_after='; ls /home/arm/junyang/carbon/data/fable_corpus/replays/ | wc -l
echo -n 'carago_replays_after='; ls /home/arm/junyang/carbon/data/carago_corpus/replays/ | wc -l
echo '--- BUILD-TOOLING-RECON ---'
ls /home/arm/junyang/carbon/code/ | grep -iE 'build|index|scan|dataset|m3s2|train' | head -20
grep -rln "TOP_TEACHERS" /home/arm/junyang/carbon/code/ 2>/dev/null | head -5
ls /home/arm/junyang/carbon/code/*.sh /home/arm/junyang/carbon/*.sh 2>/dev/null
echo '--- M3S2-BUILD-TRACE ---'
ls /home/arm/junyang/carbon/data/m3s2_corpus/ /home/arm/junyang/carbon/data/ds_m7bs/ 2>&1 | head -20
grep -rln "cnet2_build" /home/arm/junyang/carbon/code/ --include='*.py' 2>/dev/null | head -5
echo "R1FABLE1533K2 DGX-END"
DGXEOF
echo "$MARK SSH_RC=$? L20-END"
