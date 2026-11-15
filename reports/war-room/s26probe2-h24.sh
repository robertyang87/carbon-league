echo S26P2H24_BEGIN
date "+%F %T %Z"
echo "--- du subdirs ---"
timeout 120 du -sh /root/carbon/code /root/carbon/data /root/carbon/data/weights /root/carbon/data/bigeval_live 2>/dev/null
ls -d /root/carbon/.venv 2>/dev/null || echo NO_VENV_AT_CARBON
ls -d /root/.venv /root/venv 2>/dev/null
which python3; python3 -c "import sys; print(sys.executable)"
echo "--- find exp ---"
timeout 60 find /root/carbon -maxdepth 3 -name "echo2.py" 2>/dev/null
timeout 60 find /root/carbon -maxdepth 2 -name "exp" -type d 2>/dev/null
timeout 30 md5sum /root/carbon/code/exp/echo2.py 2>/dev/null || true
echo "--- code top ---"
timeout 30 ls /root/carbon/code | head -30
echo "--- weights count/size ---"
timeout 30 ls /root/carbon/data/weights | wc -l; timeout 60 du -sh /root/carbon/data/weights
echo S26P2H24_END
