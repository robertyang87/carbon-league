"""The r4 reviewer's exact X01 acceptance geometry, plus the two
added cases (occupancy conflict, shared budget). Reuses build_repro's
Feat/Prefix root builder.

ds_r4 roots:
  root0 (reviewer's case): step 200, cash 40, two bases; the record's
    stored features are the FIRST arm's prefix (recruit at slot 10,
    stay at 11); candidates = [(recruit, stay), (stay, recruit)] --
    each legal under its own prefix; under the stored prefix the
    second arm's slot-11 recruit is masked (40 - 30 < 30).
  root1 (occupancy conflict): cash 500, two ADJACENT collectors at
    cells 80/81 (slots 0/1). Logged: both stay. Candidate: slot 0
    moves RIGHT onto cell 81 -- when slot 1's features are computed,
    reserved[81] is 1 under the candidate's own prefix, 0 under the
    logged prefix (slot 1's own advance has not happened yet either
    way); slot 1's mf[*][0] at destination 81 differs accordingly.
  root2 (shared budget): cash 70; logged (stay, stay) at the bases;
    candidate (recruit, recruit): the second recruit is legal only
    because the shared pool is debited exactly once by the first
    (70 - 30 = 40 >= 30), and its uf cash column at slot 11 must
    carry the post-debit value, not the root's.
ds_r4a: root0 alone, for the exact CE comparison against the
per-candidate replay reference.
Reference Prefix features for every (root, arm) go to ref_r4.npz.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import build_repro  # noqa: E402
from build_repro import build_root, seal_dataset, make_parent, stay_except  # noqa: E402
import cnet2_feat as F  # noqa: E402


def obs_adjacent(step, cash):
    own_workers = {
        "w0": [80, 30.0, "COLLECTOR"],    # (5,5) -> slot 0
        "w1": [81, 20.0, "COLLECTOR"],    # (5,6) -> slot 1 (adjacent)
        "w2": [137, 10.0, "COLLECTOR"],   # (9,2)
        "w3": [112, 0.0, "PLANTER"],      # (7,7)
        "w4": [53, 0.0, "PLANTER"],       # (3,8)
    }
    players = [[cash, {"b0": 64, "b1": 160}, own_workers, {}],
               [500.0, {"x0": 70, "x1": 154},
                {"e0": [100, 15.0, "COLLECTOR"]}, {}]]
    return {"step": step, "players": players,
            "carbon": np.full(F.N, 40.0), "trees": {}}


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    parent_md5 = make_parent(os.path.join(outdir, "parent.npz"))
    refs = {}

    row0, arms0, refs0, y0 = build_root(
        200, 40.0, stay_except({10: 1, 11: 0}),
        [{}, {10: 0, 11: 1}])
    orig_obs = build_repro.make_obs
    build_repro.make_obs = obs_adjacent
    try:
        row1, arms1, refs1, y1 = build_root(
            201, 500.0, stay_except({}),
            [{}, {0: 2}, {0: 2, 1: 2}])
    finally:
        build_repro.make_obs = orig_obs
    row2, arms2, refs2, y2 = build_root(
        202, 70.0, stay_except({10: 0, 11: 0}),
        [{}, {10: 1, 11: 1}, {10: 1, 11: 0}])
    roots = [
        {"row": row0, "arms": arms0, "Q": [0.55, 0.62],
         "priors": [0.45, 0.55], "game": 0, "step": 200, "seat": 0},
        {"row": row1, "arms": arms1, "Q": [0.50, 0.57, 0.44],
         "priors": [0.4, 0.35, 0.25], "game": 0, "step": 201, "seat": 0},
        {"row": row2, "arms": arms2, "Q": [0.52, 0.60, 0.46],
         "priors": [0.4, 0.35, 0.25], "game": 1, "step": 202, "seat": 0},
    ]
    gid = seal_dataset(os.path.join(outdir, "ds_r4"), parent_md5, roots)
    seal_dataset(os.path.join(outdir, "ds_r4a"), parent_md5, roots[:1])
    for i, rfs in enumerate((refs0, refs1, refs2)):
        for k, (u, m, l) in enumerate(rfs):
            refs[f"r{i}_k{k}_uf"] = u; refs[f"r{i}_k{k}_mf"] = m
            refs[f"r{i}_k{k}_legal"] = l
    np.savez(os.path.join(outdir, "ref_r4.npz"), **refs)
    print(json.dumps({"parent_md5": parent_md5, "gen": gid,
                      "y_logs": [y0.tolist(), y1.tolist(), y2.tolist()],
                      "arms1": arms1.tolist()}))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "r4"))
