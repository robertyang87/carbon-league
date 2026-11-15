echo K4PROBE2002_BEGIN
date "+%F %T %Z"
echo "--- FULL EXP LIST ---"
ls /root/carbon/code/exp/
echo "--- PK_K4R1 ---"
cat /root/carbon/code/exp/pk_k4r1.py 2>/dev/null || echo NO-PK-K4R1
echo "--- M3S2R1 WRAPPER ---"
cat /root/carbon/code/exp/m3s2r1.py 2>/dev/null | head -80 || echo NO-M3S2R1
echo "--- K4R1 WEIGHTS ---"
ls -l /root/carbon/data/weights/ | grep -i -E "k4r1|k4v2" || echo NO-K4-WEIGHTS
md5sum /root/carbon/data/weights/cnet2_k4r1L.npz 2>/dev/null || echo NO-K4R1L
echo "--- DEFAULTS IN CNET2_SEARCH ---"
grep -n "MAX_BR\|BUDGET\|MIN_DEPTH\|SEARCH" /root/carbon/code/cnet2_search.py | head -30
echo K4PROBE2002_END
