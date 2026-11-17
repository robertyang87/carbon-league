"""smoke_datagen.py -- construction smokes for distill_datagen (stage 2).

Generates NO corpus data. Three checks:
  1. exclusion checker against the REAL Tier-2 slice file (read-only load):
     correct md5 accepted; candidate list containing slice ids + appendix
     ids is filtered to overlap 0; a wrong expected md5 refuses (SystemExit);
     --only-games naming an excluded game refuses.
  2. shard mechanics: two positions from a freshly seeded FastCarbon (NOT
     from any corpus game), labelled by the np-batch labeller with the real
     teacher weights, written as a shard and verified field-by-field.
  3. labels sanity: t_labels == argmax over legal-masked t_logits per slot
     (masked slots -1), t_u in [0,1].

Run on the seat that holds the pinned stack, e.g.:
  python smoke_datagen.py --code-dir /path/to/pkg/code \
      --teacher-weights /path/to/cnet2_m3s2r1_heads_v2.npz \
      --slice-file /path/to/small_teacher_tier2_slice50.json \
      --slice-md5 8f946da7c6f8c5767459506c5c2b6563 --tmp-dir /tmp/smoke_datagen
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distill_datagen as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--teacher-weights", required=True)
    ap.add_argument("--slice-file", required=True)
    ap.add_argument("--slice-md5", required=True)
    ap.add_argument("--tmp-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.tmp_dir, exist_ok=True)
    out = {"checks": []}

    # ---- check 1: exclusion gate -------------------------------------
    slice_ids = D.load_id_file(a.slice_file)
    cands = slice_ids[:3] + ["selfplay-994400", "994401", "clean-a", "clean-b"]
    excluded, rec = D.load_exclusions(a.slice_file, a.slice_md5,
                                      ["994400", "994401"], [])
    kept, crec = D.check_candidates(cands, excluded)
    assert kept == ["clean-a", "clean-b"], kept
    assert crec["overlap_after_filter"] == 0 and crec["n_excluded_present"] == 5, crec
    out["checks"].append({"exclusion_filter": crec, "slice_record": rec})
    try:
        D.load_exclusions(a.slice_file, "0" * 32, ["994400"], [])
        raise AssertionError("wrong md5 was accepted")
    except SystemExit:
        out["checks"].append({"wrong_md5_refused": True})
    try:
        D.check_candidates(cands, excluded, only_games=[slice_ids[0]])
        raise AssertionError("excluded --only-games was accepted")
    except SystemExit:
        out["checks"].append({"only_games_excluded_refused": True})
    print("check 1 (exclusion gate): PASS")

    # ---- check 2+3: shard mechanics + label sanity --------------------
    sys.path.insert(0, a.code_dir)
    import cnet2_feat as F
    from fastsim import FastCarbon
    from eval_v53 import CFG
    import cnet2_search as S
    from cnet2_search import CachePolicy

    sim = FastCarbon(CFG).reset(seed=123)
    positions = []
    for seat in (0, 1):
        feat = F.Feat(seat)
        x = feat.observe(sim.obs(seat))
        positions.append({"x": {k: np.array(x[k], copy=True) for k in D.X_FIELDS},
                          "cash": x["cash"], "slots": x["slots"], "left": x["left"],
                          "aux": {k: np.array(feat.aux[k], copy=True) for k in D.AUX_FIELDS},
                          "step": 0, "seat": seat})
    weights = dict(np.load(a.teacher_weights))
    pol = CachePolicy(weights)
    lab = D.label_np_batch({"search": S}, pol, positions, teacher_style=2, batch=2)
    shard = os.path.join(a.tmp_dir, "shard_smoke.npz")
    code_md5s = {"smoke": "construction-only"}
    smd5 = D.write_shard(shard, "smoke-game", "replay", positions, lab, 2, code_md5s)
    n = D.verify_shard(shard, expect_n=2)
    d = np.load(shard, allow_pickle=False)
    logits, labels, us = d["t_logits"], d["t_labels"], d["t_u"]
    for i, p in enumerate(positions):
        for s in range(12):
            if p["x"]["tok_mask"][s]:
                assert int(labels[i, s]) == int(np.argmax(logits[i, s])), (i, s)
            else:
                assert int(labels[i, s]) == -1, (i, s)
    assert np.all((us >= 0) & (us <= 1))
    # np-batch labels must equal the production batch=1 act() choices
    for i, p in enumerate(positions):
        x = dict(p["x"]); x["cash"] = p["cash"]; x["slots"] = p["slots"]; x["left"] = p["left"]
        lab1, _lg, _v, _u = pol.act(x, p["aux"], teacher=2, want_u=True)
        assert np.array_equal(lab1, labels[i]), (lab1, labels[i])
    out["checks"].append({"shard": {"path": shard, "md5": smd5, "n_positions": n},
                          "labels_match_batch1_act": True,
                          "t_u_range": [float(us.min()), float(us.max())]})
    print("check 2+3 (shard mechanics + labels vs batch=1 act): PASS")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
