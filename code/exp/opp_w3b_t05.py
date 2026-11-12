"""opp_w3b sampled: replica W3 best epoch, all units sampled at temperature 0.5 for the whole game (exam diagnostic: training-like conditions)."""
import os, sys
_C = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'OPEN_T': 0.5, 'OPEN_STEPS': 300,
        'OPEN_KINDS': None, 'OPEN_SEED': 7, 'TEACHER': 2, 'TEACHER_LATE': 2, 'LATE_STEP': 240}
_W = os.path.join(_C, "..", "data", "weights", "cnet2_m3w3b.npz"); _AG = None
def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_W)
    return _AG(observation, configuration)
