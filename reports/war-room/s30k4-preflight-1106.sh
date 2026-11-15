MARK=S30K4PRE1106
echo "$MARK PAYLOAD-BEGIN"
date "+%F %T %Z"
echo "--- machine ---"
nproc; free -m | head -2; df -h /root | tail -1
echo "--- venv ---"
ls /root/carbon/.venv/bin/python 2>&1; /root/carbon/.venv/bin/python --version 2>&1
echo "--- weights ---"
md5sum /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz /root/carbon/data/weights/cnet2_k4v2g1_fromstudio.npz 2>&1
echo "--- harness incl build ---"
md5sum /root/carbon/code/bigeval.py /root/carbon/code/eval_v53.py /root/carbon/code/cnet2_search.py /root/carbon/code/cnet2_build.py 2>&1
echo "--- opponents ---"
ls -la /root/carbon/code/exp/opp_master2r.py /root/carbon/code/exp/echo2.py 2>&1
echo "--- residual procs ---"
ps -eo pid,ppid,etime,args | grep -E 'bigeval|snaploop|sync' | grep -v grep || echo NO-RESIDUAL-PROC
echo "--- crontab ---"
crontab -l 2>&1 | head -10
echo "--- snaploop files on box ---"
ls /root/carbon/data/*snap* /root/carbon/*snap* 2>&1 | head
echo "--- live dir (k4 tags) ---"
ls -la /root/carbon/data/bigeval_live/ | grep -i k4 || echo NO-K4-LIVE-FILES
echo "$MARK PAYLOAD-END"
