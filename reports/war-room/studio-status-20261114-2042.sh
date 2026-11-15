echo "MARKER STUDIOSTAT2042Q8R7"
hostname; uptime
echo "--- MEM-DISK ---"
df -h / | tail -1
vm_stat | head -8
echo "--- GATE-PROCS ---"
ps -o pid,etime,pcpu,pmem,command -p 72802,72804 2>/dev/null || echo "PIDS 72802/72804 NOT FOUND"
echo "--- WORKERS ---"
pgrep -f 'spawn_mai[n]' | wc -l
ps -eo pid,etime,time,pcpu,comm | grep 'spawn_main' | grep -v grep | awk '{print $1, $2, $3, $4}' | head -5
echo "TOTAL-CPU-S $(ps -eo time,comm | grep 'spawn_main' | grep -v grep | awk -F: '{s+=$1*3600+$2*60+$3} END {print s}')"
echo "--- OLD-RUN-RESIDUE ---"
ps -p 51489,51491 2>/dev/null | tail -n +2 | wc -l
echo "--- LOAD ---"
uptime
echo "--- STAGEB_Q DIR ---"
ls -l ~/junyang/carbon/data/stageB_q/q_du_formal.log ~/junyang/carbon/data/stageB_q/q_du_formal.json 2>&1
ls -lt ~/junyang/carbon/data/stageB_q 2>/dev/null | head -6
find ~/junyang/carbon/data/stageB_q -type f -mmin -20 2>/dev/null | head -5; echo "FRESH20-DONE"
date '+%Y-%m-%d %H:%M:%S %Z'
echo "MARKER STUDIOSTAT2042Q8R7 END"
