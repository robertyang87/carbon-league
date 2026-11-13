"""Scout1's policy with the search off (training-data opponent): awr round-1 net, slot 2 -> 5 at 100 -> 7 at 240."""
import os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
_W = os.path.join(_C, "..", "data", "weights")
CONF = dict(MODE="mcts", SEARCH=False, GUARD=True, GUARD_NARROW=True, MIX=True, TEACHER=2, TEACHER_LATE=5, LATE_STEP=100,
            TEACHER_LATE2=7, LATE_STEP2=240, STYLES=(0, 1, 2, 3, 4, 5, 6, 7), ID_PRIOR=0.0)


def make():
    return _S.make_search_agent(dict(CONF), weights=os.path.join(_W, "cnet2_b128awr_r1.npz"), rweights=os.path.join(_W, "cnet2_s64v8.npz"))


agent = make()
