"""ropen_dg5: rule_open M2/M4-style opening (planters + recruits, steps < 100) on top of exp/cnet2_dg5e5s57.py; cfg {}"""
import os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import rule_open as _R


def make_agent():
    return _R.make_from(os.path.join(_C, "exp", "cnet2_dg5e5s57.py"), {'THR': 80.0, 'NEAREST': True, 'SEIZE': True, 'EPL_W': 0.0, 'PRICE_CHECK': True, 'CARS_FIRST': True, 'CS': [(0, 1), (10, 3), (25, 4), (50, 7), (75, 9), (100, 10)], 'PS': [(8, 1), (25, 3), (50, 10), (75, 16), (100, 20)]})


agent = make_agent()     # eval loads exp modules fresh per game and calls module-level agent()
