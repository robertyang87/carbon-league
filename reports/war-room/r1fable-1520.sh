#!/bin/bash
# War-room probe: (1) md5-close the Studio distill r1 file gate on DGX;
# (2) pull Fable corpus+matches onto DGX and copy into data/ for staging;
# (3) recon the index/dataset build tooling for the Fable replica.
# Marker: R1FABLE1520X9K (claim only output containing this marker)
MARK=R1FABLE1520X9K
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "R1FABLE1520X9K DGX-BEGIN $(date '+%F %T %Z')"
echo '--- R1-MD5 ---'
md5sum /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz
ls -l /home/arm/junyang/carbon/data/weights/cnet2_distill_r1_fromstudio.npz
echo 'EXPECTED 93061c69d6bc7f9f3d0468f60ce54974'
echo '--- GPU-IDLE-CHECK ---'
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
ps -eo pid,etime,args | grep -E 'cnet2_train' | grep -v grep || echo NO-TRAIN-PROC
echo '--- REPO-PULL ---'
cd ~/junyang/carbon-repo && git rev-parse --abbrev-ref HEAD && git fetch origin 2>&1 | tail -2 && git checkout cnet2-search-v12 2>&1 | tail -1 && git pull --ff-only origin cnet2-search-v12 2>&1 | tail -3 && git log --oneline -2
echo '--- FABLE-FILES ---'
ls data/fable_corpus/ | wc -l
ls -l data/fable_matches.json
echo '--- FABLE-COPY-TO-DATA ---'
mkdir -p /home/arm/junyang/carbon/data/fable_corpus/replays
cp -n ~/junyang/carbon-repo/data/fable_corpus/*.json.gz /home/arm/junyang/carbon/data/fable_corpus/replays/
cp -n ~/junyang/carbon-repo/data/fable_matches.json /home/arm/junyang/carbon/data/fable_matches.json
echo -n 'replays_copied='
ls /home/arm/junyang/carbon/data/fable_corpus/replays/ | wc -l
echo '--- BUILD-TOOLING-RECON ---'
ls /home/arm/junyang/carbon/code/ | grep -iE 'build|index|scan|dataset|m3s2|train' | head -20
grep -rln "TOP_TEACHERS" /home/arm/junyang/carbon/code/ 2>/dev/null | head -5
ls /home/arm/junyang/carbon/code/*.sh /home/arm/junyang/carbon/*.sh 2>/dev/null
echo '--- M3S2-BUILD-TRACE ---'
ls -l /home/arm/junyang/carbon/data/m3s2_corpus/ /home/arm/junyang/carbon/data/ds_m7bs/ 2>&1 | head -20
grep -rln "cnet2_build" /home/arm/junyang/carbon/code/ --include='*.py' 2>/dev/null | head -5
echo "R1FABLE1520X9K DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
