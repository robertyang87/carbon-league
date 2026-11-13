"""ropen_dg5: rule_open M2/M4-style opening (planters + recruits, steps < 100) on top of exp/cnet2_dg5e5s57.py; cfg {}"""
import os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import rule_open as _R


def make_agent():
    return _R.make_from(os.path.join(_C, "exp", "cnet2_dg5e5s57.py"), {})


agent = make_agent()     # eval loads exp modules fresh per game and calls module-level agent()
