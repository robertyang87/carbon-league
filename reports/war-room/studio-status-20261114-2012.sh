echo "MARKER STUDIOSTAT2012KX9A"
hostname; uptime
df -h / | tail -1
echo "--- launcher/gate ---"
ps -o pid,etime,pcpu,comm -p 72802,72804 2>/dev/null || echo "PIDS 72802/72804 NOT FOUND"
pgrep -f 'stage1_teacher_gat[e]' | wc -l
echo "--- workers ---"
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,pcpu,comm | grep 'spawn_main' | grep -v grep | awk '{print $1, $2, $3, $4}' | head -25
echo "--- q gate files ---"
ls -la ~/junyang/carbon/data/q_du_formal.log ~/junyang/carbon/data/q_du_formal.json 2>&1
ls -lt ~/junyang/carbon/data/stageB_q 2>/dev/null | head -5
echo "--- old run residue ---"
ps -p 51489,51491 2>/dev/null | tail -n +2 | wc -l
date '+%Y-%m-%d %H:%M:%S %Z'
