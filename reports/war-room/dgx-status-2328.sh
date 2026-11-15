#!/bin/bash
# DGX status probe 2328: GPU idle 6th trip check, HEAD vs main count, corpus counts, git residue, Fable g1 weight presence.
# Read-only. Marker: DGXSTATUS2328A
MARK=DGXSTATUS2328A
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 240 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
echo "DGXSTATUS2328A DGX-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROCS ---'
ps -eo pid,etime,args | grep -E 'cnet2_train|rl2_ppo|train|play_batch|bigeval' | grep -v grep || echo NO-TRAIN-PROC
echo '--- GPU ---'
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo '--- GIT-RESIDUE ---'
ps -eo pid,etime,args | grep -E 'git (fetch|pull|checkout)' | grep -v grep || echo NO-GIT-PROC
echo '--- HEAD-LOCAL ---'
cd /home/arm/junyang/carbon-repo && timeout 60 git rev-parse HEAD && timeout 60 git rev-parse --abbrev-ref HEAD && timeout 60 git status --porcelain=v1 | head -5; echo HEAD-RC=$?
echo '--- HEAD-REMOTE ---'
cd /home/arm/junyang/carbon-repo && REMOTE_MAIN=$(timeout 60 git ls-remote origin main | cut -f1); echo "remote-main=$REMOTE_MAIN"; echo -n "behind-count="; timeout 60 git rev-list --count HEAD..$REMOTE_MAIN 2>/dev/null || echo UNKNOWN
echo '--- FABLE-G1-WEIGHT ---'
ls -l --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/cnet2_fable_g1_fromstudio.npz 2>/dev/null && md5sum /home/arm/junyang/carbon/data/weights/cnet2_fable_g1_fromstudio.npz 2>/dev/null || echo NO-G1-WEIGHT
echo '--- WEIGHTS-NEWEST ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/weights/*.npz 2>/dev/null | head -8
echo '--- CORPUS-COUNTS ---'
cd /home/arm/junyang/carbon
for c in m3s2_corpus carago_corpus fable_corpus m9xj_corpus m9xj320_corpus; do
  echo -n "$c idx="; python3 -c "import json;print(len(json.load(open('data/$c/index.json'))))" 2>/dev/null || echo -n MISSING; echo -n " replays="; ls data/$c/replays 2>/dev/null | wc -l
done
echo '--- DS-META ---'
for d in ds_m9xj320 ds_m9xj320_w ds_carago ds_fable; do echo -n "$d: "; ls -d data/$d 2>/dev/null && python3 -c "import json;m=json.load(open('data/$d/meta.json'));print(m.get('num_samples') or m.get('n') or list(m)[:6])" 2>/dev/null || echo MISSING; done
echo '--- RECENT-FILES-30M ---'
find /home/arm/junyang/carbon/data /home/arm/junyang/carbon-repo/data -mmin -30 -type f 2>/dev/null | head -10 || echo NONE-RECENT
echo '--- SMOKE-ARTIFACTS ---'
ls -lt --time-style=+%F_%T /home/arm/junyang/carbon/data/smoke* /home/arm/junyang/carbon/smoke* 2>/dev/null | head -5; find /home/arm/junyang/carbon -maxdepth 2 -mmin -120 -type f -newer /home/arm/junyang/carbon/data/weights/cnet2_m9xj320wr1_fromstudio.npz 2>/dev/null | head -10
echo '--- LOADAVG ---'
cat /proc/loadavg
echo "DGXSTATUS2328A DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
