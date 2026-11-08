"""Stand-in (practice, not held-out) v2 for Master2: b128v7 fine-tuned on ALL its ladder games (slot 5). Never in training/search pools.
Never use in training pools or as a search opponent-model component."""
import os, sys
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _D)
import cnet2_agent as CA
_AG = CA.make_agent(os.path.join(_D, "..", "data", "weights", "cnet2_m2v2stand.npz"), teacher=5)
def agent(obs, cfg): return _AG(obs, cfg)
