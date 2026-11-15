echo S26FINAL_BEGIN
date "+%F %T %Z"; hostname
echo "--- MD5 CHECKLIST on spot-26 ---"
timeout 120 md5sum /root/carbon/code/exp/echo2.py /root/carbon/code/exp/echo_e0.py /root/carbon/code/eval_v53.py /root/carbon/code/bigeval.py /root/carbon/data/weights/cnet2_m3w5L.npz /root/carbon/data/weights/cnet2_m3w3L.npz /root/carbon/data/weights/cnet2_s64.npz 2>&1
echo "--- DU ---"
timeout 60 du -sh /root/carbon /root/carbon/code /root/carbon/.venv /root/carbon/data/weights 2>/dev/null
timeout 30 ls /root/carbon/data/weights | wc -l
echo "--- rsync active on target? ---"
pgrep -a rsync || echo NO_RSYNC_ON_TARGET
echo "--- VENV SANITY ---"
timeout 60 /root/carbon/.venv/bin/python -c "import sys, numpy; print('py', sys.version.split()[0], 'numpy', numpy.__version__)" 2>&1 | head -3
cd /root/carbon/code && timeout 120 /root/carbon/.venv/bin/python -m py_compile bigeval.py eval_v53.py exp/echo2.py exp/echo_e0.py && echo PY_COMPILE_OK
echo S26FINAL_END
