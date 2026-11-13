"""b128dg3: style 5 (Master2) until step 240, then style 7 (1C-Zero-v11), narrowed guard, no search"""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'RW': 'cnet2_s64_q8.npz', 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}
_WTS = os.path.join(_C, "..", "data", "weights", 'cnet2_b128dg3.npz')
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    if observation["step"] == 299 and os.environ.get("CS_STATS"):
        st = dict(_AG.stats); st["tag"] = 'v6s57_240'
        open(os.environ["CS_STATS"], "a").write(json.dumps(st) + "\n")
    return a
