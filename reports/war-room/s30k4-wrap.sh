echo S30K4WRAP_BEGIN
date "+%F %T %Z"
ls /root/carbon/data/bigeval_live/
cat > /root/carbon/code/exp/k4v2r1.py <<'PYEOF'
"""k4v2r1: K4TacSearch-v2 attention replica (cnet2_k4v2r1_fromstudio.npz), SEARCH ON.
K4 probe arm: identical SEARCH config to exp/k4r1_on.py (pack-frozen), weights swapped to the v2 attention replica."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_k4v2r1_fromstudio.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
PYEOF
/root/carbon/.venv/bin/python -m py_compile /root/carbon/code/exp/k4v2r1.py && echo PYCOMPILE-OK
md5sum /root/carbon/code/exp/k4v2r1.py /root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz
grep -h "CONF = " /root/carbon/code/exp/k4v2r1.py
/root/carbon/.venv/bin/python -c "
import numpy as np
d = np.load('/root/carbon/data/weights/cnet2_k4v2r1_fromstudio.npz')
ks = list(d.files)
print('k4v2r1 weight keys=', len(ks), 'meta' in ks, ks[:3])
"
echo S30K4WRAP_END
