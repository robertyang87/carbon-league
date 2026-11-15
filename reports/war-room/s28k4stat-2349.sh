echo S28K4STAT2349_BEGIN
date "+%F %T %Z"
cat /root/carbon/data/k4probe.status 2>/dev/null || echo NO-STATUS
echo "bigeval count: $(pgrep -fc 'bigeval.py exp/' || echo 0)"
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null
tail -1 /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null | grep -o '"t": "[^"]*"' | head -1
grep -h '"tag": "k4v2' /root/carbon/data/bigeval.jsonl 2>/dev/null | cut -c1-400 || echo NO-SUMMARIES-YET
cat /proc/loadavg
echo S28K4STAT2349_END
