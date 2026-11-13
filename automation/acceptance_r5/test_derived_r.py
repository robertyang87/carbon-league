"""Final acceptance: derived-layer consumption (set_uw -> sidecar -> cnet2_train.load)."""
import json, os, shutil, subprocess, sys

CODE = os.path.expanduser("/home/hatch/workspace/fix4-residual/tree")
WORK = os.path.expanduser("~/workspace/final-accept/work")
PY = os.path.expanduser("~/workspace/integ-accept/venv/bin/python")
sys.path.insert(0, CODE)
os.environ["BC_DEVICE"] = "cpu"
import numpy as np

import cnet2_feat as F
from cnet2_build import FIELDS

N = 6
PASS = []
def chk(name, cond, info=""):
    PASS.append((name, bool(cond), info))
    print(("PASS " if cond else "FAIL ") + name, info)

def build_ds(path, with_gen=True):
    if os.path.exists(path): shutil.rmtree(path)
    os.makedirs(path)
    rng = np.random.default_rng(0)
    for k, (dt, shp) in FIELDS.items():
        if k == "gid": arr = np.zeros(N, np.int64)
        elif k == "y": arr = rng.integers(0, 5, (N, F.T)).astype(dt)
        elif k == "tok_kind": arr = rng.integers(0, 2, (N, F.T)).astype(dt)
        elif k == "step": arr = np.arange(N).astype(dt)
        else: arr = np.zeros((N,) + shp, dt)
        np.save(os.path.join(path, k + ".npy"), arr)
    json.dump({"games": [{"val": 0}], "total": N}, open(os.path.join(path, "meta.json"), "w"))
    if with_gen:
        gd = os.path.join(path, "generations", "gen-test1"); os.makedirs(gd)
        man = {"generation": "gen-test1", "total": N,
               "fields": {k: {"file": k + ".npy", "rows": N, "sha256": "x"} for k in FIELDS}}
        json.dump(man, open(os.path.join(gd, "manifest.json"), "w"))
        json.dump({"generation": "gen-test1", "dir": "generations/gen-test1"},
                  open(os.path.join(path, "current.json"), "w"))
    return path

def run_set_uw(ds, *args):
    env = dict(os.environ, PYTHONPATH=CODE)
    return subprocess.run([PY, os.path.join(CODE, "set_uw.py"), ds, *map(str, args)],
                          capture_output=True, text=True, env=env)

def try_load(ds):
    import io, contextlib
    import cnet2_train as T
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            D, is_val, meta = T.load(ds)
        return ("ok", D, meta, buf.getvalue())
    except SystemExit as e:
        return ("exit", None, str(e), buf.getvalue())
    except Exception as e:
        return ("err", None, f"{type(e).__name__}: {e}", buf.getvalue())

# --- 1. happy path: real set_uw registers, real load verifies + weights enter loss formula
ds = build_ds(os.path.join(WORK, "ds1"))
r = run_set_uw(ds, 0.0, 2.0, 1.0)
chk("set_uw exit0", r.returncode == 0, r.stdout.strip()[-120:] + r.stderr[-200:])
side = json.load(open(os.path.join(ds, "derived.json")))
chk("sidecar format+base", side["format"] == "cnet2-derived/1" and
    all(v["base_generation"] == "gen-test1" for v in side["files"].values()), str(list(side["files"])))
st, D, meta, out = try_load(ds)
chk("load verified mode", st == "ok" and meta["derived_layer_check"]["mode"] == "verified",
    str(meta.get("derived_layer_check") if st == "ok" else meta))
if st == "ok":
    w_file = np.load(os.path.join(ds, "w.npy")); uw_file = np.load(os.path.join(ds, "uw.npy"))
    chk("load w/uw == set_uw files", np.array_equal(np.asarray(D["w"]), w_file) and np.array_equal(np.asarray(D["uw"]), uw_file))
    # losses() weighting, verbatim: wv = valid*tw[:,None]*uw ; lpi = (ce*wv).sum()/wv.sum()
    y = np.asarray(D["y"]); valid = (y >= 0).astype(np.float32)
    ce = np.full(y.shape, 0.7, np.float32)
    tw = np.asarray(D["w"], np.float32)
    wv = valid * tw[:, None] * np.asarray(D["uw"], np.float32)
    lpi_loaded = float((ce * wv).sum() / max(wv.sum(), 1.0))
    wv2 = valid * w_file[:, None].astype(np.float32) * uw_file
    lpi_manual = float((ce * wv2).sum() / max(wv2.sum(), 1.0))
    chk("loss weighting equality", abs(lpi_loaded - lpi_manual) < 1e-9, f"{lpi_loaded} vs {lpi_manual}")
    chk("weight_gate prod from these weights", meta["weight_gate"]["prod_n"] == int(valid.size), str(meta["weight_gate"]))

# --- 2. tamper base_generation in BOTH sidecar and mirror -> base gen rejection
ds2 = build_ds(os.path.join(WORK, "ds2")); run_set_uw(ds2, 0.0, 2.0, 1.0)
for fn in ("derived.json",):
    s = json.load(open(os.path.join(ds2, fn)))
    for v in s["files"].values(): v["base_generation"] = "gen-old"
    json.dump(s, open(os.path.join(ds2, fn), "w"))
m = json.load(open(os.path.join(ds2, "meta.json"))); m["derived_layer"] = json.load(open(os.path.join(ds2, "derived.json")))
json.dump(m, open(os.path.join(ds2, "meta.json"), "w"))
st, _, msg, _ = try_load(ds2)
chk("tamper base_generation rejected", st == "exit" and "base_generation" in msg, msg[:140])

# --- 3. tamper rows in both -> rows rejection
ds3 = build_ds(os.path.join(WORK, "ds3")); run_set_uw(ds3, 0.0, 2.0, 1.0)
s = json.load(open(os.path.join(ds3, "derived.json")))
for v in s["files"].values(): v["rows"] = 999
json.dump(s, open(os.path.join(ds3, "derived.json"), "w"))
m = json.load(open(os.path.join(ds3, "meta.json"))); m["derived_layer"] = s
json.dump(m, open(os.path.join(ds3, "meta.json"), "w"))
st, _, msg, _ = try_load(ds3)
chk("tamper rows rejected", st == "exit" and "rows" in msg, msg[:140])

# --- 4. flip bytes in uw.npy -> sha256 rejection
ds4 = build_ds(os.path.join(WORK, "ds4")); run_set_uw(ds4, 0.0, 2.0, 1.0)
uw = np.load(os.path.join(ds4, "uw.npy")); uw2 = uw.copy(); uw2[0, 0] = uw2[0, 0] + 1
np.save(os.path.join(ds4, "uw.npy"), uw2)
st, _, msg, _ = try_load(ds4)
chk("tamper file bytes rejected (sha256)", st == "exit" and "sha256" in msg, msg[:140])

# --- 5. mirror-only tamper -> mirror mismatch rejection
ds5 = build_ds(os.path.join(WORK, "ds5")); run_set_uw(ds5, 0.0, 2.0, 1.0)
m = json.load(open(os.path.join(ds5, "meta.json"))); m["derived_layer"]["files"]["w.npy"]["rows"] = 12345
json.dump(m, open(os.path.join(ds5, "meta.json"), "w"))
st, _, msg, _ = try_load(ds5)
chk("mirror mismatch rejected", st == "exit" and "mirror" in msg, msg[:140])

# --- 6. set_uw pointed INTO a generation dir -> guard refusal
r = run_set_uw(os.path.join(ds, "generations", "gen-test1"), 0.0, 2.0, 1.0)
chk("set_uw into generation refused", r.returncode != 0 and "immutable" in (r.stderr + r.stdout), (r.stderr or r.stdout)[-160:])

# --- 7. legacy dataset: no pointer/sidecar, w/uw by hand -> load ok + loud warning
ds7 = build_ds(os.path.join(WORK, "ds7"), with_gen=False)
np.save(os.path.join(ds7, "w.npy"), np.ones(N, np.float32))
np.save(os.path.join(ds7, "uw.npy"), np.ones((N, F.T), np.float32))
st, D, meta, out = try_load(ds7)
chk("legacy load ok + warning", st == "ok" and meta["derived_layer_check"]["mode"] == "legacy-unverified"
    and "WARNING: derived identity unverified" in out, str(meta.get("derived_layer_check") if st == "ok" else meta))

# --- 8. generation dataset with UNDECLARED w.npy (stale hazard) -> refusal
ds8 = build_ds(os.path.join(WORK, "ds8"))
np.save(os.path.join(ds8, "w.npy"), np.ones(N, np.float32))
st, _, msg, _ = try_load(ds8)
chk("undeclared stale w refused", st == "exit" and "neither" in msg, msg[:140])

bad = [p for p in PASS if not p[1]]
print(f"\n== derived-layer: {len(PASS) - len(bad)}/{len(PASS)} passed ==")
sys.exit(1 if bad else 0)
