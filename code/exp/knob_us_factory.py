"""US_FACTORY for data/knob_search.py: Grove1's policy (m2t3 net, slot 2 -> 7 at 240) with the search OFF; make(**conf) overrides the
cnet2_search config (e.g. MACRO=...). Fresh agent per call."""
import os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
_W = os.path.join(_C, "..", "data", "weights")
CONF = dict(MODE="mcts", SEARCH=False, GUARD=True, GUARD_NARROW=True, MIX=True, TEACHER=2, TEACHER_LATE=7, LATE_STEP=240,
            STYLES=(0, 1, 2, 3, 4, 5, 6, 7), ID_PRIOR=0.0)


def make(**over):
    return _S.make_search_agent(dict(CONF, **over), weights=os.path.join(_W, "cnet2_b128m2t3_q8.npz") if os.path.exists(os.path.join(_W, "cnet2_b128m2t3_q8.npz")) else os.path.join(_W, "cnet2_b128m2t3.npz"),
                                rweights=os.path.join(_W, "cnet2_s64mm.npz"))
