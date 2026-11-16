"""Acceptance 1: same weights on both arms -> every du exactly 0.0,
summary delta_u exactly 0.0; and a full rerun reproduces every row
bitwise (determinism). Roots: steps 285/260 (late window, short
continuations). Run from the stage dir."""
import json
import sys

sys.path.insert(0, ".")
import k0_duel as D  # noqa: E402

PARENT = "/home/hatch/workspace/heads-v2-diff/dgx.npz"
specs_all, _ = D.load_roots("../fixtures/roots.json", 200, 299)
specs = [s for s in specs_all if s["step"] in (285, 260)]
canon = sorted(({"game": s["game"], "step": s["step"], "me": s["me"],
                 "replay_md5": s["replay_md5"]} for s in specs),
               key=lambda r: (r["game"], r["step"], r["me"]))
import hashlib
sha = hashlib.sha256(json.dumps(canon, sort_keys=True,
                                separators=(",", ":")).encode()).hexdigest()

runs = []
for tag in ("run1", "run2"):
    res = D.run_duel(PARENT, PARENT, specs, sha)
    D._atomic_dump(D._sanitize(res), f"../evidence/f1_{tag}.json")
    runs.append(res)

for res in runs:
    for r in res["rows"]:
        assert r["du"] == 0.0, f"du not exactly 0: {r}"
        assert r["u_student"] == r["u_parent"], f"arms differ: {r}"
    assert res["summary"]["delta_u"] == 0.0, res["summary"]["delta_u"]
rows1 = [(r["game"], r["step"], r["u_student"], r["u_parent"], r["du"])
         for r in runs[0]["rows"]]
rows2 = [(r["game"], r["step"], r["u_student"], r["u_parent"], r["du"])
         for r in runs[1]["rows"]]
assert rows1 == rows2, "rerun not bitwise identical"
print("F1 PASS: all du == 0.0 exactly; delta_u == 0.0; rerun bitwise "
      f"identical; u values: {[r[2] for r in rows1]}")
