#!/bin/bash
# H23 prestage step D: create exp/m3s2g1.py from m3s2r1pre.py template (weights filename only), syntax-check.
# Does NOT launch any exam. Does NOT touch the running m3s2r1pre exam.
MARK=PREH23D7K9
echo "$MARK H23-BEGIN $(date '+%F %T %Z')"
hostname
cd /root/carbon/code || exit 9
echo '--- CLONE WRAPPER ---'
sed 's/cnet2_m3s2r1_preL\.npz/cnet2_m3s2g1L.npz/g' exp/m3s2r1pre.py > exp/m3s2g1.py
ls -l exp/m3s2g1.py
echo '--- DIFF template vs new (expect only weights filename) ---'
diff exp/m3s2r1pre.py exp/m3s2g1.py || true
echo '--- NEW WRAPPER CONTENT ---'
cat exp/m3s2g1.py
echo '--- SYNTAX CHECK ---'
../.venv/bin/python -c "import ast; ast.parse(open('exp/m3s2g1.py').read()); print('SYNTAX_OK')"
echo '--- WEIGHTS LOAD CHECK ---'
../.venv/bin/python -c "import numpy as np; d=np.load('/root/carbon/data/weights/cnet2_m3s2g1L.npz'); print('NPZ_KEYS', len(d.files))"
md5sum /root/carbon/data/weights/cnet2_m3s2g1L.npz
echo '--- EXAM STILL RUNNING (read-only) ---'
ps -eo pid,etime,args | grep -E 'bigeval.py exp/m3s2r1pre' | grep -v grep | head -3 || echo EXAM-PROC-NOT-SEEN
echo "$MARK H23-END"
