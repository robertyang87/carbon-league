"""Acceptance #2: independent q recomputation.

Loads a sealed annotation generation through k0_annotate.load_annotated
(verify-or-refuse), picks roots, and recomputes Q / q / priors / arms by
calling stage1_teacher_gate.extract_scene + stage0_teacher.evaluate_root
DIRECTLY (not through k0_annotate's evaluation wrapper), using the
teacher config recorded in the generation's own manifest. Comparison is
bitwise (np.array_equal on the exact stored dtypes).

Usage: python3 recompute_check.py OUT_DIR GAME_ID[,GAME_ID...] N_ROOTS
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/run")

import numpy as np  # noqa: E402

import k0_annotate as A  # noqa: E402
import stage0_teacher as T  # noqa: E402
import stage1_teacher_gate as G  # noqa: E402


def main():
    out_dir, games_want, n_pick = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
    meta, arrs = A.load_annotated(out_dir)
    cfg = meta["teacher"]["config"]
    weights = meta["teacher"]["weights"]
    games = {g["game"]: g for g in meta["games"]}
    G._W["weights_path"] = weights
    G._init_worker(weights)
    pol = G._W["pol"]
    checked = 0
    results = []
    for gi, gmeta in enumerate(meta["games"]):
        if gmeta["game"] not in games_want:
            continue
        idx = np.nonzero(arrs["gid"] == gmeta["gid"])[0]
        for i in idx:
            if checked >= n_pick:
                break
            step = int(arrs["step"][i]); seat = int(arrs["seat"][i])
            spec = {"game": gmeta["game"],
                    "path": os.path.join(sys.argv[4], gmeta["game"] + ".json.gz")
                    if len(sys.argv) > 4 else None,
                    "step": step, "me": seat, "ft": 299}
            if spec["path"] is None:
                raise SystemExit("replay dir argument required")
            scene = G.extract_scene(spec)
            r = T.evaluate_root(G._fresh_sim(scene), scene["me"], pol,
                                scene["x_me"], scene["aux_me"],
                                scene["x_opp"], scene["aux_opp"],
                                G._ns(scene["fm"]), G._ns(scene["fo"]),
                                teacher=cfg["teacher_style"],
                                K=cfg["K"], M=cfg["M"], H=cfg["H"],
                                seed_base=cfg["seed_teacher"] + step,
                                feats=(scene["fm"], scene["fo"]),
                                guard_snaps=G._teacher_guard_snaps(
                                    scene, cfg["belief"]),
                                belief=cfg["belief"])
            G._expect_teacher(r, cfg["belief"])
            n = int(arrs["n_arms"][i])
            ok = {
                "n_arms": n == len(r["arms"]),
                "arms": bool(np.array_equal(
                    arrs["arms"][i, :n].astype(np.int64),
                    np.stack([np.asarray(a, dtype=np.int64) for a in r["arms"]]))),
                "Q": bool(np.array_equal(arrs["Q"][i, :n],
                                         np.asarray(r["Q_mean"], dtype=np.float64))),
                "Q_std": bool(np.array_equal(arrs["Q_std"][i, :n],
                                             np.asarray(r["Q_std"], dtype=np.float64))),
                "q": bool(np.array_equal(arrs["q"][i, :n],
                                         np.asarray(r["q"], dtype=np.float64))),
                "priors": bool(np.array_equal(arrs["priors"][i, :n],
                                              np.asarray(r["priors"], dtype=np.float64))),
            }
            results.append({"game": gmeta["game"], "step": step, "seat": seat,
                            "bitwise": ok,
                            "all": all(ok.values())})
            checked += 1
    print(json.dumps({"checked": checked, "results": results}, indent=1))
    if not results or not all(r["all"] for r in results):
        raise SystemExit("RECOMPUTE MISMATCH")
    print("RECOMPUTE: all roots bitwise identical")


if __name__ == "__main__":
    main()
