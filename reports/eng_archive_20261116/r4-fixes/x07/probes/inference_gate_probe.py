"""X07 inference-gate proof against the FIXED driver: build the
bhead-NaN policy DIRECTLY (CachePolicy over a doctored dict, i.e.
downstream of any load gate) and run one continuation. Pre-fix this
returned U bit-identical to the healthy run (0.16894850197364647)
despite NaN logits rows; the fixed driver must reject the
measurement at inference, naming the non-finite policy output.
Run from the stage2 dir."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402
from cnet2_search import CachePolicy  # noqa: E402

man = json.load(open("../fixtures/roots_x07.json"))
spec = {"game": man[0]["game"], "path": man[0]["path"], "step": 285, "me": 0}
scene = D.extract_scene(spec)
pol_nan = CachePolicy(dict(np.load("../fixtures/corrupt_bhead.npz")))
pol_ok = D._load_policy("/home/hatch/workspace/heads-v2-diff/dgx.npz")
try:
    u = D.continue_game_2pol(pol_nan, pol_ok, scene, 995400 + 285,
                             where="probe root")
    print(f"RESULT: continuation ACCEPTED NaN-logits policy, U = {u!r} (BUG)")
except SystemExit as e:
    print(f"RESULT: SystemExit: {e}")
