"""cnet2_search_mkexp -- write an unpacked eval wrapper exp/cnet2_search_TAG.py for a search config.
python3 cnet2_search_mkexp.py TAG "{'H': 5, ...}" [WEIGHTS]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
T = '''"""unpacked cnet2_search wrapper (local eval only): %(tag)s"""
import json, os, sys
_C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _C)
import cnet2_search as _S
CONF = %(conf)s
_WTS = os.path.join(_C, "..", "data", "weights", %(w)r)
_AG = None


def agent(observation, configuration):
    global _AG
    if _AG is None or observation["step"] == 0:
        _AG = _S.make_search_agent(CONF, weights=_WTS)
    a = _AG(observation, configuration)
    if observation["step"] == 299 and os.environ.get("CS_STATS"):
        st = dict(_AG.stats); st["tag"] = %(tag)r
        open(os.environ["CS_STATS"], "a").write(json.dumps(st) + "\\n")
    return a
'''
tag, conf = sys.argv[1], sys.argv[2]
w = sys.argv[3] if len(sys.argv) > 3 else "cnet2_b128_q8.npz"
p = os.path.join(HERE, "exp", "cnet2_search_%s.py" % tag)
open(p, "w").write(T % dict(tag=tag, conf=conf, w=w)); print(p)
