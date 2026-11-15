MARK=S30K4LOC0426
echo "$MARK PAYLOAD-BEGIN"
date "+%F %T %Z"
echo "bigeval-procs:"; pgrep -fc "bigeval.py" || true
uptime
echo "--- k4v2r1 artifacts on spot-30 ---"
ls -la /root/carbon/code/exp/k4v2r1.py /root/carbon/code/exp/k4v2g1.py 2>&1
md5sum /root/carbon/code/exp/k4v2r1.py 2>/dev/null
ls -la /root/carbon/data/weights/ | grep -i "k4v2" || echo NO-K4V2-WEIGHTS
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz 2>/dev/null
echo "--- harness (post Phase C) ---"
md5sum /root/carbon/code/bigeval.py /root/carbon/code/eval_v53.py /root/carbon/code/cnet2_search.py 2>/dev/null
ls /root/carbon/code/exp/ | grep -E "opp_master2r|echo2" 
ls /root/carbon/data/bigeval_live/ 2>/dev/null | tail -5
echo "$MARK PAYLOAD-END"
