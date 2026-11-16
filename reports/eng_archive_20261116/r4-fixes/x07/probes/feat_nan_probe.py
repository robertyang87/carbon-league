"""Feature-NaN probe: extract a real root scene, poison the root
seat's step-0 feature tensors (scene['x_me']) with NaN, and run one
bare-net continuation with HEALTHY canonical policies on both seats.
Records what U comes back (pre-fix expectation: a number is returned
with no complaint -- the corrupt features silently steer the root
seat's play). Run from the stage dir."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

PARENT = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
specs, _ = D.load_roots("../fixtures/roots_x07.json", 200, 299)
spec = specs[0]
scene = D.extract_scene(spec)

x = scene["x_me"]
poisoned = []
for k, v in x.items():
    if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
        v[...] = np.nan
        poisoned.append(k)
print(f"poisoned x_me fields: {poisoned}")

pol = D._load_policy(PARENT)
try:
    u = D.continue_game_2pol(pol, pol, scene, 995400 + spec["step"])
    print(f"continue_game_2pol returned U = {u!r} "
          f"(finite={u == u and abs(u) != float('inf')})")
    print("RESULT: continuation ACCEPTED NaN features and returned a "
          "value with no error")
except SystemExit as e:
    print(f"RESULT: SystemExit: {e}")
