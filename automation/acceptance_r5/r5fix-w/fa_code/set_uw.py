"""set_uw DS [CAR_W [PLANTER_W [BASE_W [END_W [END_FROM]]]]] -- per-unit weights for a teacher dataset whose cars are NOT to be learned:
uw = PLANTER_W (1) for planter tokens, BASE_W (1) for base tokens, CAR_W (0) for collector tokens; w = 1 for every row,
or END_W for rows with step >= END_FROM (default 250): the teacher's endgame, e.g. Master3s' last 50 steps.
(With BC_SELF_KL in cnet2_train the collectors are held on the initial net instead.)
The recruit mix (base tokens) is learned from a handful of games, the planters' movement is not: a net that has only
the first half recruits fewer planters and loses its tree income. Weight the planters up and check the ledger."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnet2_build import (assert_not_in_generation, load_generation,  # noqa: E402
                         register_derived)

ds = sys.argv[1]; car_w = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ds_abs = os.path.abspath(ds)
assert_not_in_generation(ds_abs, "set_uw write weights into")
# P1-B (owner re-review): pin and VERIFY the base generation before
# touching any feature, and refuse a torn publish outright.
# BUILD_FAILED.json in the root means an interrupted publish left the
# flat mirror mixing generations -- the old code read tok_kind/y/step
# from that mirror and registered the result against the CURRENT
# pointer, silently binding one build's weights to another build's
# generation while the loader reported "verified".
_bf = os.path.join(ds_abs, "BUILD_FAILED.json")
if os.path.exists(_bf):
    sys.exit(f"set_uw: {_bf} is present -- the dataset root is a torn flat "
             f"mirror left by an interrupted publish; rebuild the dataset "
             f"before deriving weights (refusing to run)")
try:
    gen_dir, _man = load_generation(ds_abs, verify=True)
    base_gen = _man["generation"]
    feat_dir = gen_dir          # derivation inputs: certified generation ONLY
except FileNotFoundError:
    # Pre-generation legacy dataset: no pointer exists, the flat root
    # is the only source there is (registration stays impossible and
    # the consumer will flag the weights unverified, as before).
    gen_dir = None; base_gen = None; feat_dir = ds_abs
pl_w = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0; base_w = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
kind = np.load(os.path.join(feat_dir, "tok_kind.npy"), mmap_mode="r"); y = np.load(os.path.join(feat_dir, "y.npy"), mmap_mode="r")
N, S = y.shape
uw = np.full((N, S), base_w, np.float32)
uw[:, :10] = np.where(np.asarray(kind[:, :10]) == 0, car_w, pl_w)
end_w = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0; end_from = int(sys.argv[6]) if len(sys.argv) > 6 else 250
w = np.ones(N, np.float32)
if end_w != 1.0:
    w[np.load(os.path.join(feat_dir, "step.npy")) >= end_from] = end_w
np.save(os.path.join(ds, "uw.npy"), uw); np.save(os.path.join(ds, "w.npy"), w)
# Derived-layer registration (owner pt.4 / final acceptance): w/uw are
# post-processing products, never generation fields -- register them in
# the dataset root sidecar so cnet2_train.load can verify their identity
# (base generation, rows, sha256) before consuming them. The pinned
# base is passed as expected_base so a pointer that moved since the
# pin is refused inside register_derived BEFORE the sidecar is
# written; afterwards the binding is re-resolved once more, and a
# late move rolls the sidecar back to its previous bytes.
if gen_dir is not None:
    _sp = os.path.join(ds_abs, "derived.json")
    _prev_side = None
    if os.path.exists(_sp):
        with open(_sp, "rb") as _fh:
            _prev_side = _fh.read()
    _side = register_derived(ds_abs, [os.path.join(ds_abs, "uw.npy"), os.path.join(ds_abs, "w.npy")],
                             params={"car_w": car_w, "planter_w": pl_w, "base_w": base_w,
                                     "end_w": end_w, "end_from": end_from, "tool": "set_uw"},
                             expected_base=base_gen)
    try:
        _now = load_generation(ds_abs, verify=False)[1]["generation"]
    except Exception:
        _now = None
    if _now != base_gen:
        if _prev_side is None:
            try:
                os.remove(_sp)
            except OSError:
                pass
        else:
            with open(_sp, "wb") as _fh:
                _fh.write(_prev_side)
        sys.exit(f"set_uw: generation pointer moved during registration "
                 f"({base_gen} -> {_now}); the derived registration was "
                 f"rolled back -- rerun set_uw")
    print({"derived_layer": _side["format"],
           "base_generation": next(iter(_side["files"].values()))["base_generation"]})
else:
    print({"derived_layer": "WARNING: no generation pointer in dataset; w/uw left "
                            "UNREGISTERED (legacy dataset) -- consumers will flag "
                            "derived identity as unverified"})
v = np.asarray(y) >= 0; k = np.asarray(kind[:, :10])
print({"rows": int(N), "planter_tokens": int((v[:, :10] & (k == 1)).sum()), "car_tokens": int((v[:, :10] & (k == 0)).sum()),
       "base_tokens": int(v[:, 10:].sum()), "car_w": car_w, "planter_w": pl_w, "base_w": base_w,
       "end_w": end_w, "end_rows": int((w != 1.0).sum())})
