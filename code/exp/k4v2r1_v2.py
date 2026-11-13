"""k4v2r1 v2: K4TacSearch-v2 attention replica (cnet2_k4v2r1_fromstudio.npz), SEARCH ON.
K4 probe arm: identical SEARCH config to exp/k4r1_on.py (pack-frozen), weights swapped to the v2 attention replica.
v2 change (2026-11-13): passthrough `_AG.last_stats` onto this wrapper function, so
eval_v53's `getattr(fa, "last_stats")` sees search stats (the v1 wrapper swallowed them:
last_stats lived only on the make_search_agent object, so a_stats was always empty).
Frozen v1: code/exp/k4v2r1.py is NOT in main (spot-31 local, md5 c395faa8f3d783c509cb2f9fc82db676) -- unchanged.
CONF and weights are byte-identical to v1."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': True, 'ROLL': 'self', 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240, 'BUDGET': 0.7, 'HARD': 0.85, 'ABORT_DL': 0.95, 'START_DL': None}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_k4v2r1_fromstudio.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    out = _AG(observation, configuration)
    agent.last_stats = _AG.last_stats
    return out
