#!/usr/bin/env python3
"""behavior_stats.py — extract per-game behavior stats for the A-side strategy
from a bigeval_live per-game jsonl (Carbon League exam harness output).

Usage:
  python3 behavior_stats.py <jsonl_path> [--summary-mean M]
      [--summary-plants A,B] [--summary-dep A,B] [--summary-tree A,B]

The A/B side mapping inside each line's lx block is NOT assumed: three
candidate mappings (pol==A always; pol==A iff side==1; pol==A iff side==0)
are scored against structural evidence (lx[side]._dprev vs top-level
dep/odep) and, when provided, against the harness summary aggregates.
All numbers are computed from the file itself; nothing is estimated.
"""
import sys, json, math
from statistics import mean, median, pstdev

def pct(vals, q):
    if not vals: return None
    s = sorted(vals); k = (len(s) - 1) * q / 100.0
    f = math.floor(k); c = math.ceil(k)
    if f == c: return s[int(k)]
    return s[int(f)] * (c - k) + s[int(c)] * (k - f)

def desc(vals):
    if not vals: return {}
    return {"n": len(vals), "mean": round(mean(vals), 2), "median": round(median(vals), 2),
            "sd": round(pstdev(vals), 2) if len(vals) > 1 else 0.0,
            "p10": round(pct(vals, 10), 1), "p25": round(pct(vals, 25), 1),
            "p75": round(pct(vals, 75), 1), "p90": round(pct(vals, 90), 1),
            "min": min(vals), "max": max(vals)}

def main():
    path = sys.argv[1]
    opt = {}
    args = sys.argv[2:]
    for i in range(0, len(args) - 1, 2):
        opt[args[i].lstrip("-")] = args[i + 1]
    rows = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    out = {}
    out["file"] = path
    out["lines"] = len(rows)
    if rows:
        out["t_first"] = rows[0].get("t"); out["t_last"] = rows[-1].get("t")
        out["seed_first"] = rows[0].get("seed"); out["seed_last"] = rows[-1].get("seed")
    margins = [r["margin"] for r in rows if r.get("margin") is not None]
    out["margin_A"] = desc(margins)
    out["wins_A"] = sum(1 for m in margins if m > 0)
    out["errs"] = sum(1 for r in rows if r.get("err"))
    if "summary-mean" in opt:
        out["summary_mean_check"] = {"expected": float(opt["summary-mean"]),
                                     "computed": round(mean(margins), 2),
                                     "match": abs(mean(margins) - float(opt["summary-mean"])) < 1.0}
    # key inventory
    topkeys = set()
    polkeys = set(); oppkeys = set()
    for r in rows:
        topkeys.update(r.keys())
        lx = r.get("lx") or {}
        polkeys.update((lx.get("pol") or {}).keys())
        oppkeys.update((lx.get("opp") or {}).keys())
    out["top_keys"] = sorted(topkeys)
    out["lx_keys"] = sorted(polkeys | oppkeys)
    # candidate mappings: name -> function(row) -> (a_stats, b_stats)
    def m1(r):
        lx = r.get("lx") or {}; return lx.get("pol") or {}, lx.get("opp") or {}
    def m2(r):
        lx = r.get("lx") or {}
        return (lx.get("pol") or {}, lx.get("opp") or {}) if r.get("side") == 1 else (lx.get("opp") or {}, lx.get("pol") or {})
    def m3(r):
        lx = r.get("lx") or {}
        return (lx.get("pol") or {}, lx.get("opp") or {}) if r.get("side") == 0 else (lx.get("opp") or {}, lx.get("pol") or {})
    maps = {"pol_is_A_always": m1, "pol_is_A_iff_side1": m2, "pol_is_A_iff_side0": m3}
    # structural check: whose _dprev matches top-level dep (A) / odep (B)?
    struct = {}
    for name, fn in maps.items():
        ok_a = ok_b = tot = 0
        for r in rows:
            a, b = fn(r)
            if "_dprev" in a and "_dprev" in b and "dep" in r and "odep" in r:
                tot += 1
                if abs(a["_dprev"] - r["dep"]) < 2: ok_a += 1
                if abs(b["_dprev"] - r["odep"]) < 2: ok_b += 1
        struct[name] = {"checked": tot, "A_dprev_matches_dep": ok_a, "B_dprev_matches_odep": ok_b}
    out["mapping_structural_check"] = struct
    best_map = max(struct, key=lambda k: struct[k]["A_dprev_matches_dep"] + struct[k]["B_dprev_matches_odep"])
    out["chosen_mapping"] = best_map
    fn = maps[best_map]
    # candidate fields: means under each mapping, to locate 'plants'/'tree' vs summary
    cand_fields = ["plants", "planted", "pla_succ", "pla_ok", "treesteps", "life_n", "life_sum",
                   "colsteps", "cc100_sum", "haul_all", "haul240", "gain_all", "dep270", "seized"]
    cand = {}
    for name, f in maps.items():
        cand[name] = {}
        for fld in cand_fields:
            va = [f(r)[0][fld] for r in rows if fld in f(r)[0]]
            vb = [f(r)[1][fld] for r in rows if fld in f(r)[1]]
            if va: cand[name][fld] = [round(mean(va), 2), round(mean(vb), 2)]
        # trees list aggregates
        ta = [sum(f(r)[0]["trees"]) for r in rows if "trees" in f(r)[0]]
        tb = [sum(f(r)[1]["trees"]) for r in rows if "trees" in f(r)[1]]
        if ta: cand[name]["trees_sum"] = [round(mean(ta), 2), round(mean(tb), 2)]
        la = [f(r)[0]["trees"][-1] for r in rows if "trees" in f(r)[0] and f(r)[0]["trees"]]
        lb = [f(r)[1]["trees"][-1] for r in rows if "trees" in f(r)[1] and f(r)[1]["trees"]]
        if la: cand[name]["trees_last"] = [round(mean(la), 2), round(mean(lb), 2)]
    out["candidate_field_means_AB"] = cand
    if "summary-dep" in opt:
        ea, eb = [float(x) for x in opt["summary-dep"].split(",")]
        da = [r["dep"] for r in rows if "dep" in r]; db = [r["odep"] for r in rows if "odep" in r]
        out["summary_dep_check"] = {"expected_AB": [ea, eb],
                                    "computed_toplevel_dep_AB": [round(mean(da), 2), round(mean(db), 2)] if da else None}
    for key, skey in (("summary_plants_check", "summary-plants"), ("summary_tree_check", "summary-tree")):
        if skey in opt:
            ea, eb = [float(x) for x in opt[skey].split(",")]
            scored = []
            for name in maps:
                for fld, ab in cand[name].items():
                    scored.append((abs(ab[0] - ea) + abs(ab[1] - eb), name, fld, ab))
            scored.sort()
            out[key] = {"expected_AB": [ea, eb], "closest": scored[0][1:] if scored else None,
                        "second": scored[1][1:] if len(scored) > 1 else None}
    # main behavior report under chosen mapping
    A = [fn(r)[0] for r in rows]; B = [fn(r)[1] for r in rows]
    rep = {}
    if A and "plants" in A[0]:
        pa = [a["plants"] for a in A]; pb = [b["plants"] for b in B]
        rep["plants_A"] = desc(pa); rep["plants_B"] = desc(pb)
        buckets = {"<20": 0, "20-29": 0, "30-34": 0, "35-39": 0, "40-44": 0, "45-49": 0, ">=50": 0}
        for v in pa:
            if v < 20: buckets["<20"] += 1
            elif v < 30: buckets["20-29"] += 1
            elif v < 35: buckets["30-34"] += 1
            elif v < 40: buckets["35-39"] += 1
            elif v < 45: buckets["40-44"] += 1
            elif v < 50: buckets["45-49"] += 1
            else: buckets[">=50"] += 1
        rep["plants_A_buckets"] = buckets
        rep["plants_A_ge40_share"] = round(sum(1 for v in pa if v >= 40) / len(pa), 3)
    deps = [r["dep"] for r in rows if "dep" in r]; odeps = [r["odep"] for r in rows if "odep" in r]
    if deps: rep["dep_toplevel_A"] = desc(deps)
    if odeps: rep["dep_toplevel_B"] = desc(odeps)
    for fld in ("treesteps",):
        if A and fld in A[0]:
            rep[fld + "_A"] = desc([a[fld] for a in A]); rep[fld + "_B"] = desc([b[fld] for b in B])
    if A and "trees" in A[0]:
        rep["trees_last_A"] = desc([a["trees"][-1] for a in A if a["trees"]])
        rep["trees_last_B"] = desc([b["trees"][-1] for b in B if b["trees"]])
        rep["trees_len"] = len(A[0]["trees"])
    # endgame: score snapshots at step 270 vs final
    def has_all(s, keys): return all(k in s for k in keys)
    snap_keys = ["dep270", "cash270", "cargo270"]
    rep["has_step270_score_keys"] = bool(A) and has_all(A[0], snap_keys)
    rep["has_final_cash_cargo_keys"] = bool(A) and has_all(A[0], ["cash", "cargo"])
    gain_dep = [a["_dprev"] - a["dep270"] for a in A if "_dprev" in a and "dep270" in a]
    gain_dep_b = [b["_dprev"] - b["dep270"] for b in B if "_dprev" in b and "dep270" in b]
    if gain_dep: rep["dep_gain_last30_A_proxy"] = desc(gain_dep)
    if gain_dep_b: rep["dep_gain_last30_B_proxy"] = desc(gain_dep_b)
    for fld in ("late30", "late30_self", "late15"):
        if A and fld in A[0]:
            rep[fld + "_A"] = desc([a[fld] for a in A]); rep[fld + "_B"] = desc([b[fld] for b in B])
    out["report"] = rep
    print(json.dumps(out, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
