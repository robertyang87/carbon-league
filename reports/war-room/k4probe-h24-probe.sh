echo K4PROBE2001_BEGIN
date "+%F %T %Z"
echo "--- IDLE CHECK ---"
ps -eo pid,args | grep bigeval | grep -v grep || echo NO-BIGEVAL-PROC
echo "--- EXP DIR ---"
ls -l /root/carbon/code/exp/ | head -80
echo "--- K4 WRAPPERS ---"
for f in /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py /root/carbon/code/exp/*k4v2* /root/carbon/code/exp/*k4*; do
  [ -f "$f" ] && { echo "== $f =="; cat "$f"; }
done
echo "--- SEARCH TOGGLE HUNT ---"
grep -rn "SEARCH\|search" /root/carbon/code/exp/k4v2*.py 2>/dev/null | head -30
grep -rn "SEARCH" /root/carbon/code/bigeval.py /root/carbon/code/eval_v53.py 2>/dev/null | head -20
echo "--- WEIGHTS ---"
ls -l /root/carbon/data/weights/ 2>/dev/null | grep -i -E "k4v2|m3w3" || echo NO-K4V2-WEIGHTS-IN-DATA-WEIGHTS
find /root/carbon -maxdepth 4 -iname "*k4v2*" 2>/dev/null | head -20
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz /root/carbon/data/weights/cnet2_m3w3L.npz 2>/dev/null
echo "--- HARNESS MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/code/exp/echo_e0.py 2>/dev/null
echo "--- E0 WRAPPER ---"
cat /root/carbon/code/exp/echo_e0.py 2>/dev/null | head -60
echo "--- SYNC BRANCH STATE ---"
ls /root/carbon/data/bigeval_live/ 2>/dev/null | tail -10
cat /proc/loadavg
echo K4PROBE2001_END
