"""make_spars -- (re)create the local sparring files under code/exp/ (git-ignored).

  nocamp.py         = submission_nocamp.py (v53 + no-camp recipe)
  exp/spar_tf2.py   = nocamp + tree-heavy overrides (verified non-inert: 23 plants vs 11, seed 7000)
  exp/spar_tf3.py   = nocamp + heavier tree overrides (29 plants)
  exp/nn_<name>.py  = wrapper playing data/weights/<name>.npz with nn_agent (greedy + rule layer)

Usage: python3 make_spars.py
"""

import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evo import make_variant  # noqa: E402

SPARS = {
    "spar_tf2": dict(planter_from=2, early_planters=3, retire_from=250, plant_min=5.0,
                     plant_ratio=1.15, seize_min=4.0),
    "spar_tf3": dict(planter_from=2, early_planters=4, retire_from=250, retire_keep=2, plant_min=3.0,
                     plant_ratio=1.05, seize_min=2.0, planters_late=3),
}

NN_WRAPPER = '''import sys; sys.path.insert(0, {here!r})
import nn_agent as _A
_W = _A.load_weights({w!r})
_AG = None
def agent(o, c):
    global _AG
    if _AG is None or o['step'] == 0:
        _AG = _A.make_agent(_W)
    return _AG(o, c)
'''


def main():
    shutil.copy(os.path.join(HERE, "submission_nocamp.py"), os.path.join(HERE, "nocamp.py"))
    for name, ov in SPARS.items():
        make_variant("nocamp.py", ov, name)
    wdir = os.path.join(HERE, "..", "data", "weights")
    for f in sorted(os.listdir(wdir)):
        if f.startswith("nn_") and f.endswith(".npz"):
            with open(os.path.join(HERE, "exp", f[:-4] + ".py"), "w") as fh:
                fh.write(NN_WRAPPER.format(here=HERE, w=os.path.abspath(os.path.join(wdir, f))))
    print(sorted(os.listdir(os.path.join(HERE, "exp"))))


if __name__ == "__main__":
    main()
