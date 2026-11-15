echo "MARKER STUDIOSTAT2227B8Q2"
hostname; uptime
echo "--- MEM-DISK ---"
df -h / | tail -1
vm_stat | head -8
echo "--- GATE-PROCS ---"
ps -o pid,etime,pcpu,pmem,command -p 72802,72804 2>/dev/null || echo "PIDS 72802/72804 NOT FOUND"
ps -o pid,etime,command -p 72804 2>/dev/null | tail -1
echo "--- WORKERS ---"
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,pcpu,command | grep 'spawn_main' | grep -v grep | awk '{print $1, $2, $3, $4}' | head -5
echo "--- STAGEB_Q DIR ---"
ls -l ~/junyang/carbon/data/stageB_q/q_du_formal.log ~/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -lt ~/junyang/carbon/data/stageB_q 2>/dev/null | head -6
find ~/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head -5; echo "FRESH20-DONE"
date '+%Y-%m-%d %H:%M:%S %Z'
echo "MARKER STUDIOSTAT2227B8Q2 END"
