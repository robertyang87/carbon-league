#!/bin/bash
# Studio status probe 2026-11-14 02:27: idle watch. Last read 02:13:56 (STUDIOSTAT0212Q9Z4):
# normal idle since 00:47, load 1.49/1.69/1.72, brush 4acbbaea 8/8 md5 intact,
# pilot log 2956 B, sidecar 36/113 est, no §19 files landed, v3/12-game smoke
# await C acceptance (strategy-group re-run in progress, commit 2523235).
# This trip: verify by direct Tailscale SSH read — any train proc? spawn_main?
# logs/weights changed? brush still intact? §19 files landed? new dispatch files?
MARK=STUDIOSTAT0227X9M4
echo "$MARK BEGIN $(date '+%F %T %Z')"
timeout -k 30 120 ssh -i /home/hatch/.ssh/id_ed25519 -o IdentitiesOnly=yes \
  -o UserKnownHostsFile=/home/hatch/.ssh/known_hosts -o GlobalKnownHostsFile=/dev/null \
  -o BatchMode=yes -o ConnectTimeout=25 \
  -o ProxyCommand="nc -X connect -x hatch-egress-proxy:3130 %h %p" \
  arm@100.78.149.20 'bash -s' <<'STEOF'
echo "STUDIOSTAT0227X9M4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- TRAIN-PROC ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|train|fable|m3s2|wg1|wr1|k4v2|m9xj|320|distill|relabel|lb_|s1v3|pilot' | grep -v grep || echo NO-TRAIN-PROC
echo '--- SPAWN_MAIN ---'
pgrep -f spawn_main | wc -l
ps aux | grep spawn_main | grep -v grep | wc -l
echo '--- PILOT-LOG ---'
ls -l /Users/arm/junyang/carbon/data/pilot_du_v2.log /Users/arm/junyang/carbon/data/pilot_du_v2.progress.json 2>/dev/null || echo NO-PILOT-FILES
echo '--- SIDECAR ---'
cat /Users/arm/junyang/carbon/data/pilot_du_v2.progress.json 2>/dev/null || echo NO-SIDECAR
echo '--- V3ACCEPT ---'
ls -l ~/s1v3accept_run.log 2>/dev/null || echo NO-V3ACCEPT-LOG
echo '--- PARTIAL ---'
ls -l /Users/arm/junyang/carbon/data/*PARTIAL* 2>/dev/null || echo NO-PARTIAL-LS
echo '--- NEW-LOGS ---'
ls -lt /Users/arm/junyang/carbon/data/*.log 2>/dev/null | head -8
echo '--- NEW-WEIGHTS ---'
ls -lt /Users/arm/junyang/carbon/data/weights/ 2>/dev/null | head -6
echo '--- BRUSH-MD5 ---'
md5 -q /Users/arm/junyang/carbon/code/stage0_teacher.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/stage1_teacher_gate.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/cnet2_build.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/producer.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/rl2_ppo_train.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/rl2_ppo_cnet2.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/guided_*.py 2>/dev/null
md5 -q /Users/arm/junyang/carbon/code/measure*.py 2>/dev/null
echo '--- S19-FILES ---'
ls -l /Users/arm/junyang/carbon/code/rl2_checkpoint.py /Users/arm/junyang/carbon/code/test_review_fixes.py 2>/dev/null || echo NO-S19-NEW-FILES
echo '--- STAGE1-FORMAL ---'
ls -lt /Users/arm/junyang/carbon/stage1_formal/ 2>/dev/null | head -6 || echo NO-STAGE1-FORMAL
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -3
echo "STUDIOSTAT0227X9M4 STUDIO-END"
STEOF
echo "$MARK END rc=$?"
