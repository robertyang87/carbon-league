echo H24E2LAUNCH1955_BEGIN
date "+%F %T %Z"
echo "--- GATE MD5 ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/code/exp/echo2.py /root/carbon/code/exp/m3s2r1.py /root/carbon/data/weights/cnet2_m3s2r1L.npz
M=$(md5sum /root/carbon/code/eval_v53.py | cut -d' ' -f1)
if [ "$M" != "6d24da0eeeb28bad350d8b38f08ec3fc" ]; then echo "GATE-FAIL eval_v53 not synced: $M"; echo H24E2LAUNCH1955_END; exit 0; fi
echo "--- PRE-PROCS ---"
ps -eo pid,args | grep bigeval | grep -v grep || echo NO-BIGEVAL-PROC
cd /root/carbon
EVO_WORKERS=62 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nohup /root/carbon/.venv/bin/python code/bigeval.py exp/m3s2r1.py exp/echo2.py 999000 200 m3s2r1_echo2 > /root/carbon/data/m3s2r1_echo2.launcher.log 2>&1 &
echo "MAIN_PID $!"
sleep 30
echo "--- PROCS ---"
pgrep -fc "bigeval.py exp/m3s2r1.py exp/echo2.py" || true
ps -eo pid,ppid,etime,args | grep -E "bigeval.py exp/m3s2r1.py" | grep -v grep | head -5
ps -eo pid,ppid,etime,args | grep -E "bigeval.py exp/m3s2r1.py" | grep -v grep | wc -l
echo "--- JSONL ---"
ls -l /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl 2>/dev/null || echo NO-JSONL-YET
wc -l /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl 2>/dev/null || true
echo "--- LAUNCHER LOG ---"
cat /root/carbon/data/m3s2r1_echo2.launcher.log
echo "--- LOAD ---"
cat /proc/loadavg
echo H24E2LAUNCH1955_END
