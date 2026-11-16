#!/usr/bin/env python3
"""Compare two ab_runner outputs (orig vs dedup): per game, the full
two-seat action stream, final cash, step count and comparator telemetry
must be identical. Reports totals: games, actions compared, roots
(cmp calls), completed comparisons, mismatches with first divergence.

Usage: python3 ab_compare.py ORIG.json DEDUP.json OUT.json
"""
import json
import sys


def main():
    a = json.load(open(sys.argv[1]))
    b = json.load(open(sys.argv[2]))
    out = {"orig_pack": a["pack"], "dedup_pack": b["pack"],
           "orig_conf": a["pack_conf"], "dedup_conf": b["pack_conf"]}
    assert len(a["games"]) == len(b["games"]), "game count differs"
    tot_actions = tot_roots = tot_completed = tot_changed = tot_leaf = 0
    mismatches = []
    depth = {}
    for ga, gb in zip(a["games"], b["games"]):
        assert ga["seed"] == gb["seed"] and ga["pack_seat"] == gb["pack_seat"]
        n_act = sum(len(s) for s in ga["actions"])
        tot_actions += n_act
        tot_roots += ga["cmp"]["calls"]
        tot_completed += ga["cmp"]["completed_comparisons"]
        tot_changed += ga["cmp"]["changed_count"]
        tot_leaf += ga["cmp"]["leaf_calls"]
        for h, c in (ga["cmp"].get("depth_hist") or {}).items():
            depth[h] = depth.get(h, 0) + c
        if ga["actions"] != gb["actions"]:
            div = None
            for i, (sa, sb) in enumerate(zip(ga["actions"], gb["actions"])):
                if sa != sb:
                    div = {"step": i, "orig": sa, "dedup": sb}
                    break
            mismatches.append({"seed": ga["seed"], "kind": "actions", "first_divergence": div,
                               "len_orig": len(ga["actions"]), "len_dedup": len(gb["actions"])})
        if ga["cash"] != gb["cash"] or ga["steps"] != gb["steps"]:
            mismatches.append({"seed": ga["seed"], "kind": "cash/steps",
                               "orig": [ga["cash"], ga["steps"]], "dedup": [gb["cash"], gb["steps"]]})
        # wall-clock accumulators legitimately differ between runs; every
        # count/decision field must match exactly.
        skip = {"cost_accum_s", "phase_accum_s"}
        ca = {k: v for k, v in ga["cmp"].items() if k not in skip}
        cb = {k: v for k, v in gb["cmp"].items() if k not in skip}
        if ca != cb:
            ka = {k: ca[k] for k in ca if ca[k] != cb.get(k)}
            mismatches.append({"seed": ga["seed"], "kind": "cmp_telemetry", "orig_diff_keys": ka,
                               "dedup_vals": {k: cb.get(k) for k in ka}})
    out.update({"games": len(a["games"]), "actions_compared": tot_actions,
                "roots_cmp_calls": tot_roots, "completed_comparisons": tot_completed,
                "changed": tot_changed, "leaf_calls": tot_leaf, "depth_hist": depth,
                "mismatches": mismatches,
                "verdict": "PASS" if not mismatches else "FAIL"})
    json.dump(out, open(sys.argv[3], "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
