"""HELD-OUT test stand-in for claude-AlphaCarbon-Master2s: b128 fine-tuned on Master2s 50 ladder games (teacher slot 6).
Never use in training pools or as a search opponent-model component."""
import os, sys
_D = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _D)
import cnet2_agent as CA
_AG = CA.make_agent(os.path.join(_D, "..", "data", "weights", "cnet2_m2sstand.npz"), teacher=6)
def agent(obs, cfg): return _AG(obs, cfg)
