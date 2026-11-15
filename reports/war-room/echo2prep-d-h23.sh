#!/bin/bash
# Runs on h23 via cloud assistant. Builds exp/echo2.py + pre-stages exp/m3s2r1.py (final). Does NOT touch the running pre E0 field.
MARK="ECHO2PREPR2XH23"
echo "MARKER $MARK START"
date '+%F %T %Z'; hostname
cd /root/carbon/code
cat > exp/echo2.py <<'PYEOF'
"""E2 = Echo2 (W5L) pure net, row 2 all game, no MACRO -- fixed-config agent file for head-to-head exams (both sides in one process)."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'OPEN_T': 0.0, 'OPEN_STEPS': 0,
        'OPEN_KINDS': None, 'OPEN_SEED': 0, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_m3w5L.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
PYEOF
cat > exp/m3s2r1.py <<'PYEOF'
"""m3s2r1: fixed exam candidate = cnet2_m3s2r1L.npz, pure net, guard on, style row 2 all game, greedy. Frozen config: never change it, so exams stay comparable."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'OPEN_T': 0.0, 'OPEN_STEPS': 0,
        'OPEN_KINDS': None, 'OPEN_SEED': 0, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_m3s2r1L.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
PYEOF
echo "--- AST CHECK ---"
timeout 60 ../.venv/bin/python -c "import ast; ast.parse(open('exp/echo2.py').read()); ast.parse(open('exp/m3s2r1.py').read()); print('SYNTAX_OK')"
echo "--- DIFF ECHO2 VS ECHO_E0 (expect only docstring + _W lines) ---"
diff exp/echo_e0.py exp/echo2.py
echo "--- MD5 ---"
timeout 60 md5sum exp/echo2.py exp/m3s2r1.py
echo "--- WEIGHTS ON H23 ---"
ls -la /root/carbon/data/weights/cnet2_m3w5L.npz /root/carbon/data/weights/cnet2_m3s2r1L.npz
timeout 120 md5sum /root/carbon/data/weights/cnet2_m3w5L.npz /root/carbon/data/weights/cnet2_m3s2r1L.npz
timeout 120 ../.venv/bin/python -c "
import numpy as np
for p in ('/root/carbon/data/weights/cnet2_m3w5L.npz', '/root/carbon/data/weights/cnet2_m3s2r1L.npz'):
    d = np.load(p); print(p.split('/')[-1], 'keys', len(d.files))
"
echo "--- E0PRE FIELD STILL ALIVE ---"
ps aux | grep 'm3s2r1pre_e0' | grep -v grep | head -2
timeout 30 wc -l /root/carbon/data/bigeval_live/m3s2r1pre_e0.jsonl
echo "MARKER $MARK END"
