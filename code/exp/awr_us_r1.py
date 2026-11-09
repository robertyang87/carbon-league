"""awr_us_r1: behaviour policy = AWR round-1 net (slot 2 for steps < 100); dg5 ep5 pure net, opening sampled at AWR_TAU (env) for steps < 100 with seed AWR_SEED -- AWR data generation; was b128dg5_ep5: style 5 (Master2) until step 240, then style 7 (1C-Zero-v11), narrowed guard, no search"""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'OPEN_T': float(os.environ.get('AWR_TAU', '0')), 'OPEN_STEPS': 100, 'OPEN_KINDS': tuple(k for k in os.environ.get('AWR_KINDS', '').split(',') if k) or None, 'OPEN_SEED': int(os.environ.get('AWR_SEED', '0')), 'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False, 'RW': 'cnet2_s64_q8.npz', 'TEACHER': 2, 'TEACHER_LATE': 5, 'LATE_STEP': 100, 'TEACHER_LATE2': 7, 'LATE_STEP2': 240}
_WTS = os.path.join(_C, "..", "data", "weights", 'cnet2_b128awr_r1.npz')
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    agent.last_stats = _AG.stats
    if observation["step"] == 299 and os.environ.get("CS_STATS"):
        st = dict(_AG.stats); st["tag"] = 'v6s57_240'
        open(os.environ["CS_STATS"], "a").write(json.dumps(st) + "\n")
    return a
