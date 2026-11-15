#!/bin/bash
# War-room-sync 10:19 trip: DGX K0 B0/packs + clone status + --screen pickup check. READ-ONLY. Marker DGXK01019.
MARK=DGXK01019
LOG=/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/dgx-k0-1019-ts.log
{
echo "$MARK PROBE-BEGIN $(date '+%F %T %Z')"
timeout -k 30 180 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=30 -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o StrictHostKeyChecking=accept-new -o GlobalKnownHostsFile=/dev/null -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" arm@100.84.148.73 'bash -s' <<'STEOF'
echo "DGXK01019 DGX-BEGIN $(date '+%F %T %Z')"
echo '--- K0 TRAIN PROCS ---'
ps -eo pid,etime,args | grep '[k]0_train' | cut -c1-220 || echo NO-K0-TRAIN
echo '--- SCREEN PICKUP PROCS ---'
ps -eo pid,etime,args | grep '[s]creen' | head || echo NO-SCREEN-PROC
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv 2>&1
echo '--- B0/A0/A1 OUTS ---'
ls -l /home/arm/junyang/k0_b0_out /home/arm/junyang/k0_a0_out /home/arm/junyang/k0_a1_out 2>&1
md5sum /home/arm/junyang/k0_b0_out 2>/dev/null
echo '--- PACKS IN CLONE ---'
ls -l /home/arm/junyang/carbon-repo/code/exp/k0_*.py 2>&1
md5sum /home/arm/junyang/carbon-repo/code/exp/k0_*.py 2>/dev/null
echo '--- CLONE HEAD ---'
git -C /home/arm/junyang/carbon-repo rev-parse --short HEAD 2>&1; git -C /home/arm/junyang/carbon-repo status --porcelain 2>&1 | head -5
echo '--- REMOTE HEAD ---'
git -C /home/arm/junyang/carbon-repo ls-remote origin HEAD 2>&1
echo '--- CORPUS INDEX.JSON PRESENCE ---'
ls -l /home/arm/junyang/carbon-repo/data/m3s3z_corpus/index.json /home/arm/junyang/carbon-repo/data/lite-v8_corpus/index.json /home/arm/junyang/carbon-repo/data/k4v3_corpus/index.json /home/arm/junyang/carbon-repo/data/krypton_corpus/index.json 2>&1
echo '--- CORPUS COUNTS ---'
for c in m3s3z lite-v8 k4v3 krypton; do echo "== $c =="; ls /home/arm/junyang/carbon-repo/data/${c}_corpus 2>&1 | head -3; ls /home/arm/junyang/carbon-repo/data/${c}_corpus 2>/dev/null | wc -l; done
echo '--- ANY NEW FILES SINCE 10:02 (carbon trees) ---'
find /home/arm/junyang/carbon /home/arm/junyang/carbon-repo -type f -newermt '2026-11-14 10:02' 2>/dev/null | grep -v k0_holdout | head -20
echo '--- ANY NEW FILES SINCE 10:02 (carbon-repo only) ---'
find /home/arm/junyang/carbon-repo -type f -newermt '2026-11-14 10:02' 2>/dev/null | head -20
echo '--- HOLDOUT DIR ---'
ls /home/arm/junyang/carbon/data/k0_holdout 2>/dev/null | head -3; ls /home/arm/junyang/carbon/data/k0_holdout/replays 2>/dev/null | wc -l
echo "DGXK01019 DGX-END $(date '+%F %T %Z')"
STEOF
echo "$MARK SSH-RC=$?"
echo "$MARK PROBE-END $(date '+%F %T %Z')"
} | tee "$LOG"
