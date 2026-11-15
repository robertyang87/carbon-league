echo H24E2SEED1938_BEGIN
date "+%F %T %Z"
echo "--- SEED-LINES ---"
grep -n -i "seed" /root/carbon/code/bigeval.py | head -60
echo "--- WORKER-DISPATCH-LINES ---"
grep -n -E "EVO_WORKERS|workers|Pool|Process|imap|apply_async|range\(" /root/carbon/code/bigeval.py | head -40
echo "--- FILE-SIZE ---"
wc -l /root/carbon/code/bigeval.py
echo H24E2SEED1938_END
