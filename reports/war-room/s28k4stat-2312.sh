echo S28K4STAT2312_BEGIN
date "+%F %T %Z"
echo "--- PROCS ---"
ps -eo pid,ppid,etime,args | grep 'bigeval.py exp/' | grep -v grep
echo "bigeval count: $(pgrep -fc 'bigeval.py exp/')"
echo "--- CHAIN STATUS ---"
cat /root/carbon/data/k4probe.status 2>/dev/null || echo NO-STATUS
echo "--- LIVE FILES ---"
for f in k4v2r1_e0 k4v2g1_e0; do
  P=/root/carbon/data/bigeval_live/$f.jsonl
  if [ -e "$P" ]; then
    ls -l "$P"; stat -c "mtime=%y size=%s" "$P"; wc -l "$P"
    md5sum "$P"
    awk '{ print length($0) }' "$P" | sort -n | awk '{ a[NR]=$1 } END { if(NR>0) print "linelen min="a[1]" max="a[NR]" mean="int(s/NR); else print "EMPTY" } { s+=$1 }'
    echo "first-t:"; head -1 "$P" | grep -o '"t": "[^"]*"' | head -1
    echo "last-t:"; tail -1 "$P" | grep -o '"t": "[^"]*"' | head -1
  else
    echo "$f.jsonl ABSENT"
  fi
done
echo "--- WEIGHTS ---"
ls -l /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz
echo "--- LOAD ---"
cat /proc/loadavg; uptime
echo S28K4STAT2312_END
