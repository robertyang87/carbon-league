"""unpacked cnet2_search wrapper (local eval only): mfx"""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'MODE': 'mcts', 'CLOCK': 'cpu', 'BUDGET': 99.0, 'HARD': 99.0, 'START_DL': None, 'ABORT_DL': None, 'ROOT_S': 12, 'MINC': 20.0, 'MAX_KEY': 1, 'MAX_BR': 5, 'H': 3, 'MARGIN': 15.0, 'MIN_N': 2, 'SE_K': 1.0, 'RW': 'cnet2_s64_q8.npz', 'MIX': True}
_WTS = os.path.join(_C, "..", "data", "weights", 'cnet2_b128_q8.npz')
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    if observation["step"] == 299 and os.environ.get("CS_STATS"):
        st = dict(_AG.stats); st["tag"] = 'mfx'
        open(os.environ["CS_STATS"], "a").write(json.dumps(st) + "\n")
    return a
