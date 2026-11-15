echo H24E2SEED2_BEGIN
date "+%F %T %Z"
echo "--- BIGEVAL HEAD 1-30 ---"
sed -n '1,30p' /root/carbon/code/bigeval.py
echo "--- PYTHON ---"
which python3; python3 -c "import numpy, sys; print('numpy ok', numpy.__version__, sys.executable)" 2>&1 | head -2
ls -l /root/carbon/.venv/bin/python 2>/dev/null || echo NO-VENV-PY
echo "--- PREV LAUNCHER LOGS ---"
ls -lt /root/carbon/data/*.launcher.log 2>/dev/null | head -5
echo H24E2SEED2_END
