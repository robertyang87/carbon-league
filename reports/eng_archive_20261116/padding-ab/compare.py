"""Compare A/B row files: per-root bitwise identity on shared real roots
plus aggregate contrast. Timing fields (teacher_ms, wall_s) excluded
from identity (they are wall-clock, not values)."""
import json
import math
import sys

import numpy as np

FT = 188
VAL_FIELDS = ["du", "u_guided", "u_greedy", "teacher_calls"]


def load(p):
    d = json.load(open(p))
    return {(r["step"], r["me"]): r for r in d["rows"]}, d


def agg(rows):
    du = np.array([r["du"] for r in rows], dtype=np.float64)
    return {"n_rows": len(rows),
            "n_zero_rows": int(sum(1 for r in rows if r["du"] == 0.0)),
            "root_mean_du": float(du.mean()) if len(du) else None,
            "sigma_root_du": float(du.std(ddof=1)) if len(du) > 1 else None,
            "game_mean_du": float(du.mean()) if len(du) else None,
            "teacher_calls_total": int(sum(r["teacher_calls"] for r in rows))}


def main():
    A, da = load(sys.argv[1])
    B, db = load(sys.argv[2])
    shared = sorted(set(A) & set(B))
    a_only = sorted(set(A) - set(B))
    print(f"A rows {len(A)}, B rows {len(B)}, shared {len(shared)}, A-only {len(a_only)}")
    assert all(s < FT for s, _ in shared), "shared set must be real roots"
    assert all(s >= FT for s, _ in a_only), "A-only set must be padding roots"
    n_ident = 0
    diffs = []
    for k in shared:
        bad = {f: (A[k][f], B[k][f]) for f in VAL_FIELDS if A[k][f] != B[k][f]}
        if bad:
            diffs.append((k, bad))
        else:
            n_ident += 1
    print(f"shared real roots bitwise-identical on {VAL_FIELDS}: {n_ident}/{len(shared)}")
    for k, bad in diffs:
        print("DIFF", k, bad)
    print("\nper-root table (step, me | A du | B du | A u_g | B u_g | A u_gr | B u_gr | calls A/B):")
    for k in shared:
        print(f"  {k} | {A[k]['du']:.17g} | {B[k]['du']:.17g} | "
              f"{A[k]['u_guided']:.17g} | {B[k]['u_guided']:.17g} | "
              f"{A[k]['u_greedy']:.17g} | {B[k]['u_greedy']:.17g} | "
              f"{A[k]['teacher_calls']}/{B[k]['teacher_calls']}")
    print("\nA-only (padding) rows morphology:")
    for k in a_only:
        r = A[k]
        print(f"  {k} du={r['du']:.17g} u_guided={r['u_guided']:.17g} "
              f"u_greedy={r['u_greedy']:.17g} teacher_calls={r['teacher_calls']}")
    print("\naggregates A (all roots):", json.dumps(agg(list(A.values()))))
    print("aggregates B (real only):", json.dumps(agg(list(B.values()))))
    print("aggregates A restricted to shared roots:",
          json.dumps(agg([A[k] for k in shared])))
    json.dump({"shared": [list(k) for k in shared], "n_identical": n_ident,
               "diffs": [(list(k), {f: list(v) for f, v in bad.items()})
                         for k, bad in diffs],
               "a_only": [list(k) for k in a_only],
               "agg_A": agg(list(A.values())), "agg_B": agg(list(B.values())),
               "agg_A_shared": agg([A[k] for k in shared])},
              open("/home/hatch/workspace/padding-ab/out/comparison.json", "w"),
              indent=1)
    print("\nsaved out/comparison.json")


if __name__ == "__main__":
    main()
