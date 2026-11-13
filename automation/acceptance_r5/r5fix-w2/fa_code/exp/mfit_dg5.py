"""mfit_dg5: m_rules decision-tree M-family opening (mfit, steps < 100) on top of exp/cnet2_dg5e5s57.py; cfg {}"""
import os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import m_rules as _R


def make_agent():
    return _R.make_from(os.path.join(_C, "exp", "cnet2_dg5e5s57.py"), {"MODE": "tgt", "ECOL": 1})


agent = make_agent()     # eval loads exp modules fresh per game and calls module-level agent()
