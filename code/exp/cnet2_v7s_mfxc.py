"""time-capped: BUDGET/HARD seconds per step (env SB/SH; ship 0.5/0.75 for the 1 s platform limit). unpacked cnet2_search wrapper (local eval only): mfxn"""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'MODE': 'mcts', 'CLOCK': 'wall', 'BUDGET': float(os.environ.get('SB', '0.5')), 'HARD': float(os.environ.get('SH', '0.75')), 'START_DL': None, 'ABORT_DL': float(os.environ.get('SA', '0.9')), 'ROOT_S': 12, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'RW': 'cnet2_s64_q8.npz', 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}
_WTS = os.path.join(_C, "..", "data", "weights", 'cnet2_b128v7.npz')
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    agent.last_stats = _AG.stats          # read by eval_v53 -> per-game changed/shallow in result files
    if observation["step"] == 299 and os.environ.get("CS_STATS"):
        st = dict(_AG.stats); st["tag"] = 'mfxn'
        open(os.environ["CS_STATS"], "a").write(json.dumps(st) + "\n")
    return a
