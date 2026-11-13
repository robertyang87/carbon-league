"""AWR end-to-end chain: awr_weights (floored) -> awr_collector_anchor -> cnet2_train gate.
Codex-style extreme distribution: 200 games, one margin -10000."""
import gzip, json, os, subprocess, sys
import numpy as np

PY = sys.executable
W = os.path.dirname(os.path.abspath(__file__))
PASS = []
def ok(name, cond, extra=""):
    assert cond, f"FAIL: {name} {extra}"
    PASS.append(name); print(f"PASS {name} {extra}")

# ---- fabricate corpus + dataset ----
corpus = "/tmp/awr_corpus"; ds = "/tmp/awr_ds"
os.makedirs(corpus, exist_ok=True); os.makedirs(ds, exist_ok=True)
games = []
obs = {"players": [[0.0, True, {}, {}], [0.0, True, {}, {}]], "carbon": [0] * 225}
for s in range(50):
    margins = [100.0, 110.0, 90.0, 120.0]
    if s == 0: margins[0] = -10000.0
    for i, mg in enumerate(margins):
        key = f"g{s}_{i}"
        fn = key + ".json.gz"
        with gzip.open(os.path.join(corpus, fn), "wt") as fh:
            json.dump({"steps": [[{"observation": obs}]],
                       "info": {"scores": [max(mg, 0.0), max(-mg, 0.0)], "terminal": "timeout"}}, fh)
        games.append({"file": fn, "side": 0, "key": key, "seed": 1000 + s, "reward": mg})
with open(os.path.join(corpus, "awr.jsonl"), "w") as fh:
    for g in games: fh.write(json.dumps(g) + "\n")

NG = len(games); ROWS = NG * 3
json.dump({"games": [{"key": g["key"], "val": False} for g in games]}, open(os.path.join(ds, "meta.json"), "w"))
np.save(os.path.join(ds, "gid.npy"), np.repeat(np.arange(NG), 3))
np.save(os.path.join(ds, "y.npy"), np.zeros((ROWS, 12), np.int64))
tk = np.ones((ROWS, 12), np.int64); tk[:, 0] = 0
np.save(os.path.join(ds, "tok_kind.npy"), tk)
np.save(os.path.join(ds, "tok_mask.npy"), np.ones((ROWS, 12), bool))

env = dict(os.environ, AWR_CAR_ANCHOR="1")
r = subprocess.run([PY, os.path.join(W, "awr_weights.py"), ds, corpus], capture_output=True, text=True, env=env)
ok("AWR-01 awr_weights exits 0", r.returncode == 0, r.stderr[-300:])
rep = json.loads(r.stdout.strip().splitlines()[-1])
ok("AWR-02 extreme game floored and named", rep["floored_games"] >= 1 and "g0_0" in rep["floored_keys"], str(rep)[:220])
w = np.load(os.path.join(ds, "w.npy")); uw = np.load(os.path.join(ds, "uw.npy"))
pos = w[w > 0]
ok("AWR-03 w floored at 1e-3, all finite", abs(float(pos.min()) - 1e-3) < 1e-9 and np.isfinite(w).all()
   and np.isfinite(uw).all(), f"min_pos={pos.min()} uw_max={uw.max()}")
ok("AWR-04 car-anchor uw bounded (bad game = 1/1e-3)", abs(float(uw.max()) - 1000.0) < 0.5, f"uw_max={uw.max()}")

# ---- anchor (fixed) on the floored dataset ----
out1 = "/tmp/awr_out1"
r = subprocess.run([PY, os.path.join(W, "awr_collector_anchor.py"), ds, out1, "1.0"], capture_output=True, text=True)
ok("AWR-05 anchor exits 0 on floored ds", r.returncode == 0, r.stderr[-300:])
uw1 = np.load(os.path.join(out1, "uw.npy")); w1 = np.load(os.path.join(out1, "w.npy"))
bad_rows = np.where(np.asarray(np.load(os.path.join(ds, "gid.npy"))) == 0)[0]
prod_bad = float((w1[bad_rows[:1]].astype(np.float32)[:, None] * uw1[bad_rows[:1]])[0, 0])
ok("AWR-06 collector product == scale*mean(w)", np.isfinite(uw1).all()
   and abs(prod_bad - float(w1.mean())) / float(w1.mean()) < 0.01,
   f"prod={prod_bad:.4f} mean_w={float(w1.mean()):.4f} uw_max={float(uw1.max()):.1f}")

# ---- train gate on the anchored output ----
sys.path.insert(0, W); os.environ["BC_DEVICE"] = "cpu"
import cnet2_train as T
summ = T._check_finite_weights({"w": w1, "uw": uw1}, out1)
manual = float(np.abs(w1.astype(np.float32)[:, None] * uw1.astype(np.float32)).max())
ok("AWR-07 train gate product summary on anchored data", summ["prod_max"] == manual and np.isfinite(summ["prod_max"]),
   str(summ))

# ---- bypass: anchor on a pre-floor w.npy (tiny positive weight) ----
ds2 = "/tmp/awr_ds2"; os.makedirs(ds2, exist_ok=True)
np.save(os.path.join(ds2, "w.npy"), np.array([1.0, 3.21e-6, 0.5, 0.0], np.float32))
np.save(os.path.join(ds2, "uw.npy"), np.ones((4, 12), np.float32))
np.save(os.path.join(ds2, "tok_kind.npy"), tk[:4]); np.save(os.path.join(ds2, "tok_mask.npy"), np.ones((4, 12), bool))
out2 = "/tmp/awr_out2"
r = subprocess.run([PY, os.path.join(W, "awr_collector_anchor.py"), ds2, out2, "1.0"], capture_output=True, text=True)
uw2 = np.load(os.path.join(out2, "uw.npy"))
mean2 = float(np.load(os.path.join(ds2, "w.npy")).mean())
ok("AWR-08 bypass entry floored too (no 1e5+ uw)", r.returncode == 0 and np.isfinite(uw2).all()
   and float(uw2.max()) <= mean2 / 1e-3 * 1.001 and "floored_rows" in r.stdout,
   f"uw_max={float(uw2.max()):.1f} bound={mean2/1e-3:.1f} out={r.stdout.strip()[:160]}")

# ---- bypass: anchor refuses non-finite input, writes no uw ----
ds3 = "/tmp/awr_ds3"; os.makedirs(ds3, exist_ok=True)
np.save(os.path.join(ds3, "w.npy"), np.array([1.0, np.inf, 0.5, 0.0], np.float32))
np.save(os.path.join(ds3, "uw.npy"), np.ones((4, 12), np.float32))
np.save(os.path.join(ds3, "tok_kind.npy"), tk[:4]); np.save(os.path.join(ds3, "tok_mask.npy"), np.ones((4, 12), bool))
out3 = "/tmp/awr_out3"
r = subprocess.run([PY, os.path.join(W, "awr_collector_anchor.py"), ds3, out3, "1.0"], capture_output=True, text=True)
ok("AWR-09 anchor refuses inf w, no uw saved", r.returncode != 0 and not os.path.exists(os.path.join(out3, "uw.npy")),
   r.stderr.strip()[:120] or r.stdout.strip()[:120])

print(f"\nALL {len(PASS)} AWR CHAIN CHECKS PASS")
