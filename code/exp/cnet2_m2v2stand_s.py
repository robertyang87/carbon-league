"""Stand-in (practice, not held-out) v2 for Master2 WITH test-time search (Master2 itself searches): m2s2 net + cnet2_search r10."""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'MODE': 'mcts', 'CLOCK': 'cpu', 'BUDGET': 99.0, 'HARD': 99.0, 'START_DL': None, 'ABORT_DL': None, 'ROOT_S': 10, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'RW': 'cnet2_s64v7.npz', 'STYLES': (0, 1, 2, 3, 4, 5, 6, 7), 'ID_PRIOR': 0.0, 'MIX': True, 'MINC': 20.0, 'GUARD': True, 'GUARD_NARROW': True, 'TEACHER': 5, 'TEACHER_LATE': 7, 'LATE_STEP': 240}
_WTS = os.path.join(_C, "..", "data", "weights", 'cnet2_m2v2stand.npz')
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
