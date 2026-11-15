echo BHVPRB1752K1_BEGIN
date "+%F %T %Z"
echo "--- ls bigeval_live (m3s2r1) ---"
timeout 30 ls -la --time-style=full-iso /root/carbon/data/bigeval_live/ | grep -i "m3s2r1" || echo NO_M3S2R1_FILES
echo "--- wc -l each m3s2r1 file ---"
for f in /root/carbon/data/bigeval_live/*m3s2r1*; do [ -f "$f" ] && { printf "%s " "$f"; timeout 10 wc -l < "$f"; }; done
echo "--- head -1 of each m3s2r1 jsonl (cut 4000c) ---"
for f in /root/carbon/data/bigeval_live/*m3s2r1*.jsonl; do [ -f "$f" ] && { echo "HEAD $f"; timeout 10 head -1 "$f" | cut -c1-4000; echo; }; done
echo "--- bigeval.jsonl tail (summaries mentioning m3s2r1) ---"
timeout 20 grep -a "m3s2r1" /root/carbon/data/bigeval.jsonl | tail -5 | cut -c1-2000
echo BHVPRB1752K1_END
