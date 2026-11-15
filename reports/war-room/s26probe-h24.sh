echo S26PH24_BEGIN
date "+%F %T %Z"; hostname
echo "--- du ---"
timeout 60 du -sh /root/carbon /root/carbon/code /root/carbon/.venv /root/carbon/data 2>/dev/null
echo "--- top level ---"
timeout 30 ls /root/carbon
echo "--- checklist md5 on h24 ---"
timeout 60 md5sum /root/carbon/exp/echo2.py /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py 2>&1
timeout 60 find /root/carbon -name "cnet2_m3w5L.npz" -o -name "cnet2_m3w3L.npz" -o -name "cnet2_s64.npz" 2>/dev/null | head
for f in $(timeout 60 find /root/carbon -name "cnet2_m3w5L.npz" -o -name "cnet2_m3w3L.npz" -o -name "cnet2_s64.npz" 2>/dev/null | head -6); do timeout 30 md5sum "$f"; done
echo "--- running exam untouched check ---"
timeout 30 ls -la /root/carbon/data/bigeval_live/ 2>/dev/null | grep -i m3s2r1 | head
echo S26PH24_END
