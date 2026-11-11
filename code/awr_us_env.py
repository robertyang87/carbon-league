"""awr_us_env: our pure net for awr_gen, everything from the environment (data generation on cloud hosts).
  AWR_US_WEIGHTS  policy weights (.npz path, or a name under data/weights)          required
  AWR_US_CONF     JSON conf overrides, e.g. '{"TEACHER": 2, "TEACHER_LATE": 7, "LATE_STEP": 240}'
  AWR_TAU / AWR_SEED    sampling temperature / seed (set per game by awr_gen)
  AWR_OPEN_STEPS  sample while step < this (default 100; 300 = the whole game)
  AWR_KINDS       which units are sampled, e.g. "planter,base" (cars stay greedy); empty = all"""
import json, os, sys
_C = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = {'SEARCH': False, 'GUARD': True, 'GUARD_NARROW': True, 'MODE': 'sweep', 'MIX': False,
        'OPEN_T': float(os.environ.get('AWR_TAU', '0')), 'OPEN_STEPS': int(os.environ.get('AWR_OPEN_STEPS', '100')),
        'OPEN_KINDS': tuple(k for k in os.environ.get('AWR_KINDS', '').split(',') if k) or None,
        'OPEN_SEED': int(os.environ.get('AWR_SEED', '0'))}
CONF.update(json.loads(os.environ.get('AWR_US_CONF', '{}')))
_W = os.environ['AWR_US_WEIGHTS']
_WTS = _W if os.path.exists(_W) else os.path.join(_C, "..", "data", "weights", _W)
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    agent.last_stats = _AG.stats
    agent.inner = _AG                      # .state["lg0"] / .state["f"].own_uid: the net's logits per own slot (fork_gen)
    return a
