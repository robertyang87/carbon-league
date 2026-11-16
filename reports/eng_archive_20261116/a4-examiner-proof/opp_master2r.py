"""Exam-only opponent: claude-AlphaCarbon-Master2's public source (ladder /source/e591b77f..., md5 404114a0...), kept outside
the repo in data/public_src/master2.py (or next to this file on cloud hosts). _SEED fixed so a seed replays bit-for-bit.
Never in a training pool or a submission."""
import importlib.util, os
_H = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_H, "master2_src.py"), os.path.join(_H, "..", "..", "data", "public_src", "master2.py")):
    if os.path.exists(_p):
        break
_s = importlib.util.spec_from_file_location("master2_public", _p); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
_m._SEED = 20261111
agent = _m.agent
