#!/usr/bin/env python3
"""diag_vs_echo3 -- small local diagnostic: CONF variants of the vleaf pack vs the Echo3 pack.

Rewrites the CONF line of code/exp/vleaf_value_pack.py into temp packs and plays each
variant against code/packs/echo3_int8.py on a few seeds, printing per-game margin and the
search stats (searched / shallow = discarded by MIN_DEPTH / changed, step-time counts).
Diagnostic only: wall-clock budgets make the numbers machine- and load-dependent; seeds are
arbitrary (not from any registered block); nothing here is exam-grade.

  python3 tools/katago/diag_vs_echo3.py --seeds 6 --procs 3 \
      'value_b10_t2={"LEAF_VALUE":"value","BUDGET":1.0,"HARD":1.5,"TEACHER":2}' \
      'heur_b10_t2={"BUDGET":1.0,"HARD":1.5,"TEACHER":2}' 'bare_t2={"SEARCH":false,"TEACHER":2}'
"""
import argparse, json, multiprocessing as mp, os, statistics, sys, tempfile, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code")
OLD = 'CONF = {"LEAF_VALUE": "value"}'


def run(t):
    sys.path.insert(0, CODE)
    import eval_v53
    name, pack, seed, side = t
    t0 = time.time()
    r = eval_v53.play_game(("diag", pack, "packs/echo3_int8.py", seed, side))
    st = r.get("a_stats") or {}
    ms = st.get("ms_steps", [])
    return {"variant": name, "seed": seed, "side": side, "margin": r.get("margin"), "err": r.get("err"),
            "searched": st.get("searched"), "shallow": st.get("shallow"), "changed": st.get("changed"),
            "mean_ms": round(statistics.mean(ms)) if ms else None,
            "gt1s": sum(m > 1000 for m in ms), "gt3s": sum(m > 3000 for m in ms), "wall": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+", help="name=JSON_CONF")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=424243)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    src = open(os.path.join(CODE, "exp", "vleaf_value_pack.py")).read()
    assert src.count(OLD) == 1
    td = tempfile.mkdtemp(prefix="vleaf_diag_")
    tasks = []
    for v in a.variants:
        name, conf = v.split("=", 1)
        p = os.path.join(td, name + ".py")
        open(p, "w").write(src.replace(OLD, "CONF = " + repr(json.loads(conf))))
        tasks += [(name, p, a.seed0 + i, i % 2) for i in range(a.seeds)]
    rows = []
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for r in pool.imap_unordered(run, tasks):
            rows.append(r); print(json.dumps(r), flush=True)
    for name in dict.fromkeys(t[0] for t in tasks):
        rs = sorted((r for r in rows if r["variant"] == name), key=lambda r: r["seed"])
        m = [r["margin"] for r in rs]
        print(json.dumps({"summary": name, "margins": [round(x) for x in m], "mean": round(statistics.mean(m)),
                          "wins": sum(x > 0 for x in m),
                          "searched": sum(r["searched"] or 0 for r in rs), "shallow": sum(r["shallow"] or 0 for r in rs),
                          "changed": sum(r["changed"] or 0 for r in rs), "gt1s": sum(r["gt1s"] for r in rs),
                          "gt3s": sum(r["gt3s"] for r in rs)}))


if __name__ == "__main__":
    main()
