echo H24E2RATE2002_BEGIN
date "+%F %T %Z"
echo "--- PROCS ---"
pgrep -fc "bigeval.py exp/m3s2r1.py exp/echo2.py" || true
echo "--- JSONL ---"
wc -l /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl
head -1 /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('first:', d.get('seed'), d.get('side'), d.get('margin'), d.get('t'), 'err:', d.get('err'))"
tail -1 /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('last:', d.get('seed'), d.get('side'), d.get('margin'), d.get('t'), 'err:', d.get('err'))"
echo "--- WORKER RSS ---"
WP=$(ps -eo pid,ppid --no-headers | awk '$2==88687{print $1; exit}'); echo "WORKER_PID $WP"; ps -o pid,rss,etime,args -p $WP 2>/dev/null || echo NO-WORKER-FOUND
echo "--- LOAD/MEM ---"
cat /proc/loadavg
free -m | head -2
echo H24E2RATE2002_END
