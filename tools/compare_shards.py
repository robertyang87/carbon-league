"""compare_shards -- per-root reconciliation of two annotation shard
sets (batch-2 validation, design S6.2 frame A caliber; metrics follow
the first-stage leafbatch compare: priors/q/Q/Q_std max|d|, arms diff,
argmax flips named per root, n_act/guard equality where carried).

Two input forms:
  --shards-a DIR --shards-b DIR : k0_annotate shard directories
      (OUT/shards). Per game: the embedded _meta "arrays" sha256 maps
      are compared entry by entry (per-array sha reconciliation) and
      self-checked against recomputation; per root (joined on
      (game, step, seat)): max|d| of priors/q/Q/Q_std over the root's
      recorded arms, padding-region equality (q/priors zero, Q NaN),
      arms-row differences, argmax(q) flips named, belief/n_unknown
      equality. Shards carry no n_act/guard counters -- for those use
      the recon form.
  --recon-a F.jsonl --recon-b F.jsonl : first-stage recon records
      (worker_eval format); the same numeric metrics plus n_act_calls
      and guard-dict equality per root.

Step bands (design S6.2): [0,100), [100,200), [200,299], plus global.
Exit code 0 iff every compared root is exactly equal on every metric
(max|d| == 0.0, no flips, no arms diff, sha maps equal); 1 otherwise;
2 on usage/input errors. A JSON report is written with --out.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

BANDS = ((0, 100), (100, 200), (200, 299))


def _sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _array_sha(arr):
    a = np.ascontiguousarray(arr)
    return _sha256_bytes(a.dtype.str.encode()
                         + json.dumps(list(a.shape)).encode()
                         + a.tobytes())


def band_of(step):
    for lo, hi in BANDS:
        if lo <= step < hi:
            return f"[{lo},{hi})"
    return "out"


def _agg(rows):
    if not rows:
        return None
    return {
        "n": len(rows),
        "max_d_priors": max(r["d_priors"] for r in rows),
        "max_d_q": max(r["d_q"] for r in rows),
        "max_d_Q": max(r["d_Q"] for r in rows),
        "max_d_Qstd": max(r["d_Qstd"] for r in rows),
        "flips": sum(1 for r in rows if r["flip"]),
        "arms_diff_roots": sum(1 for r in rows if r["arms_diff"]),
        "belief_diff_roots": sum(1 for r in rows if not r["belief_equal"]),
        "pad_diff_roots": sum(1 for r in rows if not r["pad_equal"]),
    }


def compare_shards(dir_a, dir_b):
    rows = []
    sha_report = {}
    games_a = {f[:-4] for f in os.listdir(dir_a) if f.endswith(".npz")}
    games_b = {f[:-4] for f in os.listdir(dir_b) if f.endswith(".npz")}
    report = {"games_only_a": sorted(games_a - games_b),
              "games_only_b": sorted(games_b - games_a),
              "sha_mismatch": [], "sha_selfcheck_fail": []}
    for g in sorted(games_a & games_b):
        sides = {}
        for tag, d in (("a", dir_a), ("b", dir_b)):
            with np.load(os.path.join(d, g + ".npz")) as D:
                meta = json.loads(str(D["_meta"].item()))
                arrs = {k: np.asarray(D[k]) for k in D.files
                        if k != "_meta"}
            for k, want in meta["arrays"].items():
                if _array_sha(arrs[k]) != want:
                    report["sha_selfcheck_fail"].append(
                        {"game": g, "side": tag, "field": k})
            sides[tag] = (meta, arrs)
        ma, mb = sides["a"][0], sides["b"][0]
        sha_eq = ma["arrays"] == mb["arrays"]
        sha_report[g] = sha_eq
        if not sha_eq:
            diff_fields = [k for k in ma["arrays"]
                           if ma["arrays"].get(k) != mb["arrays"].get(k)]
            report["sha_mismatch"].append(
                {"game": g, "fields": diff_fields})
        A, B = sides["a"][1], sides["b"][1]
        keys_a = {(int(s), int(seat)): i for i, (s, seat) in
                  enumerate(zip(A["step"].tolist(), A["seat"].tolist()))}
        keys_b = {(int(s), int(seat)): i for i, (s, seat) in
                  enumerate(zip(B["step"].tolist(), B["seat"].tolist()))}
        for key in sorted(set(keys_a) | set(keys_b)):
            step, seat = key
            base = {"game": g, "step": step, "seat": seat,
                    "band": band_of(step)}
            if key not in keys_a or key not in keys_b:
                base.update({"missing": True, "d_priors": None,
                             "d_q": None, "d_Q": None, "d_Qstd": None,
                             "flip": True, "arms_diff": -1,
                             "belief_equal": False, "pad_equal": False})
                rows.append(base)
                continue
            ia, ib = keys_a[key], keys_b[key]
            na, nb = int(A["n_arms"][ia]), int(B["n_arms"][ib])
            n = na
            arms_diff = 0
            if na != nb:
                arms_diff = abs(na - nb)
            for i in range(min(na, nb)):
                if not np.array_equal(A["arms"][ia, i], B["arms"][ib, i]):
                    arms_diff += 1
            pad_eq = bool(
                (A["q"][ia, na:] == 0).all() and (B["q"][ib, nb:] == 0).all()
                and (A["priors"][ia, na:] == 0).all()
                and (B["priors"][ib, nb:] == 0).all()
                and np.isnan(A["Q"][ia, na:]).all()
                and np.isnan(B["Q"][ib, nb:]).all())
            base.update({
                "missing": False,
                "d_priors": float(np.abs(A["priors"][ia, :n]
                                         - B["priors"][ib, :n]).max()),
                "d_q": float(np.abs(A["q"][ia, :n]
                                    - B["q"][ib, :n]).max()),
                "d_Q": float(np.abs(A["Q"][ia, :n]
                                    - B["Q"][ib, :n]).max()),
                "d_Qstd": float(np.abs(A["Q_std"][ia, :n]
                                       - B["Q_std"][ib, :n]).max()),
                "flip": bool(int(np.argmax(A["q"][ia, :n]))
                             != int(np.argmax(B["q"][ib, :n]))),
                "arms_diff": int(arms_diff),
                "belief_equal": bool(int(A["belief"][ia])
                                     == int(B["belief"][ib])
                                     and int(A["n_unknown"][ia])
                                     == int(B["n_unknown"][ib])),
                "pad_equal": pad_eq})
            rows.append(base)
    report["per_game_sha_equal"] = sha_report
    return rows, report


def compare_recon(path_a, path_b):
    def load(p):
        recs = {}
        for ln in open(p):
            if ln.strip():
                d = json.loads(ln)
                recs[(d["game"], d["step"], d["seat"])] = d
        return recs
    ra, rb = load(path_a), load(path_b)
    rows = []
    for key in sorted(set(ra) | set(rb)):
        g, step, seat = key
        base = {"game": g, "step": step, "seat": seat,
                "band": band_of(step)}
        if key not in ra or key not in rb:
            base.update({"missing": True, "flip": True, "arms_diff": -1,
                         "n_act_equal": False, "guard_equal": False})
            rows.append(base)
            continue
        a, b = ra[key], rb[key]
        arms_a = [np.array(x) for x in a["arms"]]
        arms_b = [np.array(x) for x in b["arms"]]
        arms_diff = sum(0 if (i < len(arms_b) and np.array_equal(
            arms_a[i], arms_b[i])) else 1 for i in range(len(arms_a)))
        base.update({
            "missing": False,
            "d_priors": float(np.abs(np.array(a["priors"])
                                     - np.array(b["priors"])).max()),
            "d_q": float(np.abs(np.array(a["q"])
                                - np.array(b["q"])).max()),
            "d_Q": float(np.abs(np.array(a["Q_mean"])
                                - np.array(b["Q_mean"])).max()),
            "d_Qstd": float(np.abs(np.array(a["Q_std"])
                                   - np.array(b["Q_std"])).max()),
            "flip": bool(int(np.argmax(a["q"]))
                         != int(np.argmax(b["q"]))),
            "arms_diff": int(arms_diff),
            "n_act_equal": a.get("n_act_calls") == b.get("n_act_calls"),
            "guard_equal": a.get("guard") == b.get("guard")})
        rows.append(base)
    return rows, {"mode": "recon"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-a")
    ap.add_argument("--shards-b")
    ap.add_argument("--recon-a")
    ap.add_argument("--recon-b")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.shards_a and a.shards_b:
        rows, report = compare_shards(a.shards_a, a.shards_b)
        mode = "shards"
    elif a.recon_a and a.recon_b:
        rows, report = compare_recon(a.recon_a, a.recon_b)
        mode = "recon"
    else:
        ap.error("give --shards-a/--shards-b or --recon-a/--recon-b")
    flips = [f"{r['game']}:{r['step']}:{r['seat']}" for r in rows
             if r["flip"]]
    summary = {"mode": mode, "n_roots": len(rows),
               "global": _agg(rows) if mode == "shards" else None,
               "bands": {}, "flips_named": flips,
               "missing_roots": [f"{r['game']}:{r['step']}:{r['seat']}"
                                 for r in rows if r.get("missing")]}
    for lo, hi in BANDS:
        sub = [r for r in rows if r["band"] == f"[{lo},{hi})"]
        if mode == "shards":
            summary["bands"][f"[{lo},{hi})"] = _agg(sub)
        else:
            summary["bands"][f"[{lo},{hi})"] = {
                "n": len(sub),
                "max_d_q": max([r["d_q"] for r in sub], default=None),
                "flips": sum(1 for r in sub if r["flip"]),
                "nact_mismatch": sum(1 for r in sub
                                     if not r.get("n_act_equal", True)),
                "guard_diff": sum(1 for r in sub
                                  if not r.get("guard_equal", True))}
    if mode == "recon":
        summary["global"] = {
            "n": len(rows),
            "max_d_priors": max(r["d_priors"] for r in rows),
            "max_d_q": max(r["d_q"] for r in rows),
            "max_d_Q": max(r["d_Q"] for r in rows),
            "max_d_Qstd": max(r["d_Qstd"] for r in rows),
            "flips": len(flips),
            "arms_diff_roots": sum(1 for r in rows if r["arms_diff"]),
            "nact_mismatch": sum(1 for r in rows
                                 if not r.get("n_act_equal", True)),
            "guard_diff": sum(1 for r in rows
                              if not r.get("guard_equal", True))}
    summary["report"] = report
    summary["rows"] = rows
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     indent=1, sort_keys=True))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(summary, fh, indent=1)
    ok = (not flips and not summary["missing_roots"]
          and summary["global"] is not None
          and summary["global"]["max_d_priors"] == 0.0
          and summary["global"]["max_d_q"] == 0.0
          and summary["global"]["max_d_Q"] == 0.0
          and summary["global"]["max_d_Qstd"] == 0.0
          and summary["global"]["arms_diff_roots"] == 0
          and not report.get("sha_mismatch")
          and not report.get("sha_selfcheck_fail")
          and not report.get("games_only_a")
          and not report.get("games_only_b"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
