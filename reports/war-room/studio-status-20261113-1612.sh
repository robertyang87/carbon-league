#!/bin/bash
# Studio status probe 2026-11-13 16:12: routine 15-min check. Last report 15:57: Studio normal idle, code at main b7e1db92, s12test done 15:10 (A1/A2/A3 pass), q gate awaiting policy-group wiring + Codex sign-off. Check still idle, code md5s, s12test untouched, fable/distill state.
MARK=STUDIOSTAT1612X7Q4
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o UserKnownHostsFile=/root/.jishubuddy/network-checks/studio-known-hosts-20261112 arm@10.211.43.68 'bash -s' <<'STEOF'
echo "STUDIOSTAT1612X7Q4 STUDIO-BEGIN $(date '+%F %T %Z')"
hostname; uptime
echo '--- MEM-DISK ---'
df -h / | tail -1; vm_stat | head -6
echo '--- PROCS ---'
ps -ax -o pid,%cpu,%mem,etime,command | grep -E 'cnet2|distill|train|test_s12|s12test|python' | grep -v grep || echo NO-TRAIN-PROC
echo '--- CODE ---'
cd /Users/arm/junyang/carbon/code && md5 -q cnet2_build.py cnet2_train.py set_uw.py cnet2_model.py producer/selfplay_gen.py
ls -lT cnet2_build.py cnet2_train.py set_uw.py cnet2_model.py producer/selfplay_gen.py | awk '{print $6, $7, $8, $9}'
echo '--- S12TEST ---'
ls -lt /Users/arm/s12test 2>/dev/null | head -10
echo '--- FABLE-LOGS ---'
ls -lT /Users/arm/junyang/carbon/data/fable_r1.log /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null; tail -3 /Users/arm/junyang/carbon/data/fable_r1.log 2>/dev/null; tail -3 /Users/arm/junyang/carbon/data/fable_g1.log 2>/dev/null
echo '--- DISTILL-EP19 ---'
ls -lt /Users/arm/junyang/carbon/data/weights 2>/dev/null | grep -E 'ep19' | head -6
ls -lt /Users/arm/junyang/carbon/data 2>/dev/null | head -8
echo "STUDIOSTAT1612X7Q4 STUDIO-END"
STEOF
echo "$MARK L20-END rc=$?"
