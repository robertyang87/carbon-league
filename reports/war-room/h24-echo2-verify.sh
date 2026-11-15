echo H24E2VERIFY1945_BEGIN
date "+%F %T %Z"
echo "--- PROCS ---"
ps -eo pid,etime,args | grep -E "bigeval" | grep -v grep || echo NO-BIGEVAL-PROC
echo "--- CPU ---"
nproc
grep -m1 "model name" /proc/cpuinfo
grep -c "^processor" /proc/cpuinfo
echo "--- MEM ---"
free -m
grep -E "MemTotal|MemAvailable" /proc/meminfo
echo "--- THREAD-ENV ---"
env | grep -iE "OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_NUM_THREADS|VECLIB_MAXIMUM_THREADS" || echo NO-THREAD-ENV-SET
/root/carbon/.venv/bin/python -c "import numpy,sys; print('venv numpy', numpy.__version__, 'python', sys.version.split()[0])"
echo "--- MD5S ---"
md5sum /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/code/exp/echo2.py /root/carbon/data/weights/cnet2_m3s2r1L.npz
echo "--- FIELDS ---"
wc -l /root/carbon/data/bigeval_live/m3s2r1_e0.jsonl
ls -l /root/carbon/data/bigeval_live/m3s2r1_echo2.jsonl 2>/dev/null || echo NO-ECHO2-JSONL
echo H24E2VERIFY1945_END
