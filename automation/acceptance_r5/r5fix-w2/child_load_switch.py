"""Child runner for the R3 P1 reproduction: mid-load generation switch.

Imports the workdir cnet2_train and hooks its `load_generation` reference
so the FIRST resolution inside load() -- the one that pins generation
gen-AAAA -- is immediately followed by a publish of generation gen-BBBB
plus a set_uw pass for B (inline equivalent: derive from B's sealed
features, register with expected_base=gen-BBBB). load() then continues.

Deterministic interleaving of the owner's spec: loader pins A -> B is
published and B's derived weights are generated -> load continues.
Correct behaviour: refuse (A's features must never be consumed with B's
weights under a "verified" label). Bug behaviour: load succeeds,
mode=verified against B, uw for A's four collector tokens is 2.0 where
A's own derivation (car_w=0) requires 0.0.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))
sys.path.insert(0, HERE)
os.environ.setdefault("BC_DEVICE", "cpu")

import numpy as np  # noqa: E402
import fixture  # noqa: E402
import cnet2_build as B  # noqa: E402
import cnet2_train as T  # noqa: E402

ds = sys.argv[1]


def publish_b_and_derive():
    genB = os.path.join(ds, "generations", "gen-BBBB")
    os.makedirs(genB, exist_ok=True)
    A = fixture.field_arrays(seed=7)
    kind = np.array(A["tok_kind"], copy=True)
    kind[:, :10] = 1            # in B those four slots are planters
    A["tok_kind"] = kind
    fields = {}
    for k, a in A.items():
        p = os.path.join(genB, k + ".npy")
        np.save(p, a)
        fields[k] = {"file": k + ".npy", "rows": fixture.N,
                     "sha256": fixture._sha(p)}
    meta = {"games": [{"val": False, "key": "g0"}], "total": fixture.N}
    with open(os.path.join(genB, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    with open(os.path.join(genB, "manifest.json"), "w") as fh:
        json.dump({"generation": "gen-BBBB", "total": fixture.N,
                   "fields": fields}, fh)
    with open(os.path.join(ds, "current.json"), "w") as fh:
        json.dump({"generation": "gen-BBBB", "dir": "generations/gen-BBBB",
                   "total": fixture.N}, fh)
    for k in A:                 # faithful flat mirror, as a publish does
        np.save(os.path.join(ds, k + ".npy"), A[k])
    with open(os.path.join(ds, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    # set_uw for B (car_w=0, planter_w=2, base_w=1), features from B.
    y = A["y"]
    n, S = y.shape
    uw = np.full((n, S), 1.0, np.float32)
    uw[:, :10] = np.where(np.asarray(kind[:, :10]) == 0, 0.0, 2.0)
    np.save(os.path.join(ds, "uw.npy"), uw)
    np.save(os.path.join(ds, "w.npy"), np.ones(n, np.float32))
    B.register_derived(
        ds, [os.path.join(ds, "uw.npy"), os.path.join(ds, "w.npy")],
        params={"car_w": 0.0, "planter_w": 2.0, "base_w": 1.0,
                "end_w": 1.0, "end_from": 250, "tool": "set_uw"},
        expected_base="gen-BBBB")


_orig = T.load_generation
_state = {"done": False}


def hooked(dirn, verify=True):
    out = _orig(dirn, verify=verify)
    if not _state["done"]:
        _state["done"] = True
        publish_b_and_derive()
    return out


T.load_generation = hooked

try:
    D, is_val, meta = T.load(ds)
    chk = meta.get("derived_layer_check") or {}
    out = {"ok": True,
           "maps_sha": hashlib.sha256(
               np.asarray(D["maps"]).tobytes()).hexdigest(),
           "uw_row0_head": (np.asarray(D["uw"])[0, :10].astype(float).tolist()
                            if "uw" in D else None),
           "mode": chk.get("mode"),
           "check_gen": chk.get("generation"),
           "source": meta.get("dataset_source")}
except SystemExit as e:
    out = {"ok": False, "systemexit": str(e)[:400]}
except Exception as e:  # noqa: BLE001
    out = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
print("VERDICT " + json.dumps(out))
