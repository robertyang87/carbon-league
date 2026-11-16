"""Feature-NaN probe against the FIXED driver, through the real
run_duel path: extract_scene is wrapped so the root seat's step-0
float features (scene['x_me']) are poisoned with NaN right after
extraction; run_duel must die at the per-root scene gate naming the
root and the feature, before any continuation is scored.
Run from the stage dir."""
import sys

import numpy as np

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

PARENT = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
_orig_extract = D.extract_scene


def poisoned_extract(spec):
    scene = _orig_extract(spec)
    hit = []
    for k, v in scene["x_me"].items():
        if isinstance(v, np.ndarray) and v.dtype.kind in "fc":
            v[...] = np.nan
            hit.append(k)
    print(f"poisoned x_me fields: {hit}")
    return scene


D.extract_scene = poisoned_extract
specs, sha, blobs = D.load_roots("../fixtures/roots_x07.json", 200, 299)
try:
    D.run_duel(PARENT, PARENT, specs, sha, replay_blobs=blobs)
    print("RESULT: run_duel ACCEPTED NaN features (BUG)")
except SystemExit as e:
    print(f"RESULT: SystemExit: {e}")
