#!/bin/bash
MARK=S30E3PROBE0243
echo "$MARK PAYLOAD-BEGIN"
date '+%F %T %Z'
hostname
echo "--- RESIDUAL-SCAN ---"
ps aux | grep -E "bigeval|eval_v53" | grep -v grep || echo NO-EXAM-PROCS
cat /proc/loadavg
echo "--- PACKS-DIR ---"
ls -la /root/carbon/data/packs/ 2>/dev/null || echo NO-PACKS-DIR
echo "--- EXP-LIST ---"
ls /root/carbon/code/exp/ 2>/dev/null | head -80
echo "--- PACK-WRAPPER-CANDIDATES ---"
ls /root/carbon/code/exp/ 2>/dev/null | grep -iE "sb30|sb100|n1r|pack" || echo NONE-BY-NAME
for f in /root/carbon/code/exp/sb30.py /root/carbon/code/exp/sb100.py; do
  if [ -f "$f" ]; then echo "--- CAT $f ---"; cat "$f"; fi
done
echo "--- BIGEVAL-HEAD ---"
head -50 /root/carbon/code/bigeval.py
echo "--- VENV ---"
ls -l /root/carbon/.venv/bin/python 2>/dev/null || echo NO-VENV
/root/carbon/.venv/bin/python -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null || echo NUMPY-FAIL
echo "$MARK PAYLOAD-END"
