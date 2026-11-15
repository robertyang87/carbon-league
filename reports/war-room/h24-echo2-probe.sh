echo H24E2PROBE1935_BEGIN
date "+%F %T %Z"
echo "--- BIGEVAL-PROCS ---"
ps -eo pid,etime,args | grep -E "bigeval" | grep -v grep || echo NO-BIGEVAL-PROC
echo "--- E0-FIELD ---"
wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl 2>/dev/null || echo NO-E0-JSONL
grep -c CAL_ALL_DONE /root/carbon/data/m3s2r1_cal.out 2>/dev/null || echo NO-CAL-OUT
echo "--- MD5S ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/code/exp/echo2.py /root/carbon/code/exp/m3s2r1.py 2>&1
ls /root/carbon/data/weights/ 2>/dev/null | grep -i m3s2r1 || echo NO-M3S2R1-WEIGHT
md5sum /root/carbon/data/weights/cnet2_m3s2r1L.npz 2>&1
echo "--- ECHO2-FIELD-EXISTS ---"
ls -l /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl 2>/dev/null || echo NO-ECHO2-JSONL-YET
echo "--- DISK ---"
df -h /root 2>/dev/null | tail -1
echo H24E2PROBE1935_END
