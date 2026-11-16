"""X06 probe oracle: per-(game, seat) ground-truth feature rows built by
cnet2_build.build_game with a SINGLE side -- no segmentation inference
anywhere. Also emits seat-asymmetry evidence at the root steps."""
import hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/hatch/workspace/k0-round2/join/run")  # main-byte copies
import cnet2_build as B

CORPUS = "/home/hatch/workspace/gen1r_smoke"
sys.path.insert(0, "/home/hatch/workspace/k0-round2/join")
from make_build_fixture import scan_shaped_game  # fixture machinery, import only

FEATS = ("maps", "scal", "tok_pos", "tok_kind", "tok_mask", "tok_cargo",
         "hist_cell", "hist_cargo", "hist_move", "uf", "mf", "legal",
         "carbon", "teacher", "step")
ROOT_STEPS = (206, 207, 208)

idx = json.load(open(os.path.join(CORPUS, "index.json")))
entries = {e["game"]: e for e in idx if e.get("game") in (994000, 994001)}

def row_hash(arrs, i):
    h = hashlib.sha256()
    for k in FEATS:
        h.update(k.encode()); h.update(np.ascontiguousarray(arrs[k][i]).tobytes())
    return h.hexdigest()

oracle = {}
asym = []
os.makedirs(os.path.join(HERE, "oracle"), exist_ok=True)
for gid_num in (994000, 994001):
    g = scan_shaped_game(entries[gid_num], CORPUS)
    if g.get("raw_margin") is not None:
        g["raw_payload"] = {"mg": g["raw_margin"], "u": g["raw_u"],
                            "outcome": B.OUTCOME_CODES[g.get("outcome", "normal")]}
    else:
        g["raw_payload"] = None
    per_seat = {}
    for s in (0, 1):
        tids = [B.teacher_id(g["names"][s])]
        mg = [g["scores"][s] - g["scores"][1 - s]]
        shard = os.path.join(HERE, "oracle", f"g{gid_num}_s{s}.npz")
        task = (g["file"], [s], 0, tids, mg, shard, g["raw_payload"])
        gi, n, err = B.build_game(task)
        assert err is None, (gid_num, s, err)
        D = np.load(shard)
        steps = D["step"].astype(int)
        assert list(steps) == sorted(steps), (gid_num, s, "steps not ascending")
        per_seat[s] = {k: D[k] for k in FEATS}
        for t in ROOT_STEPS:
            i = int(np.nonzero(steps == t)[0][0])
            oracle[f"selfplay-{gid_num}|{s}|{t}"] = {
                "game": f"selfplay-{gid_num}", "seat": s, "step": t,
                "sha256": row_hash(per_seat[s], i), "shard": shard, "row": i}
    # asymmetry evidence at root steps: seat0 row vs seat1 row
    for t in ROOT_STEPS:
        i0 = int(np.nonzero(per_seat[0]["step"].astype(int) == t)[0][0])
        i1 = int(np.nonzero(per_seat[1]["step"].astype(int) == t)[0][0])
        diff = {}
        for k in FEATS:
            a, b = per_seat[0][k][i0], per_seat[1][k][i1]
            diff[k] = int((a != b).sum())
        asym.append({"game": f"selfplay-{gid_num}", "step": t,
                     "diff_elems": diff,
                     "scal_seat0": [float(x) for x in per_seat[0]["scal"][i0][:8]],
                     "scal_seat1": [float(x) for x in per_seat[1]["scal"][i1][:8]],
                     "value_seat0": float(per_seat[0]["teacher"][i0]) if "teacher" in per_seat[0] else None})
json.dump({"oracle": oracle, "asymmetry": asym},
          open(os.path.join(HERE, "oracle_index.json"), "w"), indent=1)
print(json.dumps({"oracle_rows": len(oracle),
                  "asym_sample": asym[0]["diff_elems"]}, indent=1))
