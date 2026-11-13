"""S1-2 A3 guard parity: extracted BankruptcyGuard == old make_search_agent closure.

Part 1: old vs new agent outputs on identical real positions (SEARCH=False,
GUARD on, production config) -- run in two processes (old/new module) and
compare the JSON dumps exactly.
Part 2: synthetic trigger cases (risky branch incl. re-decode path, recruit
branch, no-trigger) -- old closure exec'd standalone vs BankruptcyGuard.apply.
Run: <venv python> test_s12_a3.py [old|new|synth|compare]
  old/new: drive the agent over replay positions, dump JSON to guard_parity_<which>.json
  synth:   standalone old-closure vs new-class comparison (no agent)
  compare: diff the two JSON dumps
"""
import gzip
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WHICH = sys.argv[1] if len(sys.argv) > 1 else "new"
if WHICH == "old":
    sys.path.insert(0, os.path.join(HERE, "orig"))  # old cnet2_search.py wins

import cnet2_feat as F  # noqa: E402
import cnet2_search as CS  # noqa: E402  (old or new module, per WHICH)
from cnet2_search import CachePolicy  # noqa: E402
import cnet2_np as NP  # noqa: E402

CFG = dict(SEARCH=False, GUARD=True, GUARD_NARROW=True, GUARD_CASH=40.0, LOWW=2, TEACHER=1)
REPLAYS = sorted(os.path.join(HERE, "replays", f) for f in os.listdir(os.path.join(HERE, "replays"))
                 if f.endswith(".json.gz"))


def drive_agent():
    w = dict(np.load(os.path.join(HERE, "heads_v2.npz")))
    agent = CS.make_search_agent(conf=dict(CFG), weights=w, rweights=w)
    out = []
    for rp in REPLAYS[:3]:
        d = json.load(gzip.open(rp))
        st = d["steps"]
        for t in range(0, 61, 3):
            o0 = st[t][0]["observation"]
            obs = {"step": o0["step"], "players": o0["players"],
                   "carbon": st[t][0]["observation"]["carbon"], "player": 0,
                   "trees": o0.get("trees", {}), "remainingOverageTime": 0}
            acts = agent(obs)
            s = dict(agent.last_stats)
            out.append({"game": os.path.basename(rp)[:8], "t": t, "acts": acts,
                        "guard_moves": s.get("guard_moves", 0),
                        "guard_recruit": s.get("guard_recruit", 0),
                        "errors": s.get("errors", 0)})
    with open(os.path.join(HERE, f"guard_parity_{WHICH}.json"), "w") as f:
        json.dump(out, f)
    print(f"{WHICH}: {len(out)} positions, "
          f"guard_moves={sum(o['guard_moves'] for o in out)}, "
          f"guard_recruit={sum(o['guard_recruit'] for o in out)}, "
          f"errors={sum(o['errors'] for o in out)}")


def compare():
    a = json.load(open(os.path.join(HERE, "guard_parity_old.json")))
    b = json.load(open(os.path.join(HERE, "guard_parity_new.json")))
    # Hardened (Codex review P3): an empty comparison or shared non-zero
    # error counts must not be able to print PASS.
    assert len(a) == len(b) and len(a) > 0, "length mismatch or empty comparison"
    assert all(o["errors"] == 0 for o in a) and all(o["errors"] == 0 for o in b), \
        "non-zero errors in parity fixtures"
    # Fixture identity lock (Codex R2 P3): canonical fixture is 63 unique
    # (game, t) positions; duplicates or shrinkage must not pass.
    ka = [(o["game"], o["t"]) for o in a]
    kb = [(o["game"], o["t"]) for o in b]
    assert len(set(ka)) == len(ka) and len(set(kb)) == len(kb), \
        "duplicate fixture identities"
    assert len(a) >= 50, f"parity fixture shrank: {len(a)} entries"
    n_diff = 0
    for x, y in zip(a, b):
        if x != y:
            n_diff += 1
            print("DIFF", x["game"], x["t"], {k: (x[k], y[k]) for k in x if x[k] != y[k]})
    print(f"agent parity: {len(a)} positions, diffs={n_diff}")
    assert n_diff == 0, "guard extraction parity FAILED"


def synth():
    """Old closure (exec'd from the original file) vs BankruptcyGuard.apply."""
    import ast
    import textwrap
    src = open(os.path.join(HERE, "orig", "cnet2_search.py")).read()
    tree = ast.parse(src)
    fn_src = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "make_search_agent":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "bankrupt_guard":
                    fn_src = textwrap.dedent(ast.get_source_segment(src, sub))
    assert fn_src, "old bankrupt_guard not found"

    from cnet2_search import BankruptcyGuard
    w = dict(np.load(os.path.join(HERE, "heads_v2.npz")))
    pol = CachePolicy(w)
    # a real position for x/aux
    d = json.load(gzip.open(REPLAYS[0]))
    st = d["steps"]
    o0 = st[100][0]["observation"]
    obs = {"step": o0["step"], "players": o0["players"],
           "carbon": st[100][0]["observation"]["carbon"], "player": 0}
    f = F.Feat(0)
    x = f.observe(obs)
    aux = f.aux
    y0, lg0, _ = pol.act(x, aux, teacher=1)
    y0 = np.asarray(y0, dtype=np.int64)

    def old_guard(x_, aux_, y_, lg_, st_, stats_):
        ns = {"np": np, "F": F, "decode_b": __import__("cnet2_search").decode_b,
              "st": st_, "stats": stats_, "cfg": dict(CFG), "pol": pol}
        exec("def f_own_uids():\n    return [u for u in st['f'].own_uid[:10] if u is not None]\n"
             + fn_src.replace("def bankrupt_guard(x, aux, y, lg0):",
                              "def bankrupt_guard(x, aux, y, lg0):", 1), ns)
        return ns["bankrupt_guard"](x_, aux_, y_, lg_)

    cases = []
    # case 1: no trigger (plain position)
    cases.append(("plain", dict(x), aux, y0.copy(), dict(lostw=0, prevw=set())))
    # case 2: recruit branch (no collector): mask all collectors out
    x2 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in x.items()}
    for s in range(10):
        if x2["tok_mask"][s] and int(x2["tok_kind"][s]) == 0:
            x2["tok_mask"][s] = False
    x2["cash"] = 100.0
    x2["slots"] = 1  # a free base slot: the recruit branch needs slots > 0
    y2 = y0.copy()
    y2[10] = 0
    y2[11] = 0  # bases stay: force the recruit branch to fire
    cases.append(("recruit", x2, aux, y2, dict(lostw=0, prevw=set())))
    # case 3: risky branch: 2 workers left, one lost before, cash 20,
    #         and slot keep[0]'s current move is bad while move 4 is safe.
    x3 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in x.items()}
    keep = [s for s in range(10) if x3["tok_mask"][s]][:2]
    for s in range(10):
        if s not in keep:
            x3["tok_mask"][s] = False
    x3["cash"] = 20.0
    aux3 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in aux.items()}
    d = F.DEST[int(x3["tok_pos"][keep[0]])]  # 5 destination cells of slot keep[0]
    thr = np.zeros_like(aux3["thr"], dtype=np.float64)
    thr[d[:4]] = 1.0   # moves 0..3 threatened
    thr[d[4]] = 0.0    # move 4 safe
    aux3["thr"] = thr
    en_min = np.full_like(aux3["en_min"], np.inf, dtype=np.float64)
    en_min[d[:4]] = 0.0  # enemy collector can arrive carrying 0 <= our cargo
    aux3["en_min"] = en_min
    y3 = y0.copy()
    y3[keep[0]] = 0  # a bad move for slot keep[0]
    lost_uid = "player-0-worker-99"
    cases.append(("risky", x3, aux3, y3, dict(lostw=0, prevw={lost_uid})))

    for name, xx, aa, yy, init in cases:
        st_old = {"f": f, "T_now": 1, "lostw": init["lostw"], "prevw": set(init["prevw"]),
                  "guard_base": set()}
        stats_old = {}
        r_old = old_guard(xx, aa, yy.copy(), lg0, st_old, stats_old)
        g = BankruptcyGuard(dict(CFG))
        g.lostw = init["lostw"]
        g.prevw = set(init["prevw"])
        stats_new = {}
        r_new = g.apply(xx, aa, yy.copy(), lg0, pol, f.own_uid, 1, stats_new)
        same = np.array_equal(np.asarray(r_old), np.asarray(r_new))
        s_same = (stats_old.get("guard_moves", 0) == stats_new.get("guard_moves", 0)
                  and stats_old.get("guard_recruit", 0) == stats_new.get("guard_recruit", 0))
        print(f"{name}: labels_equal={same} stats_equal={s_same} "
              f"old_moves={stats_old.get('guard_moves', 0)}/{stats_old.get('guard_recruit', 0)} "
              f"new_moves={stats_new.get('guard_moves', 0)}/{stats_new.get('guard_recruit', 0)} "
              f"changed={not np.array_equal(np.asarray(r_old), yy)}")
        assert same and s_same, f"synth case {name} FAILED"
    print("A3 synth PASS (old closure == BankruptcyGuard.apply on all trigger branches)")


if __name__ == "__main__":
    if WHICH in ("old", "new"):
        drive_agent()
    elif WHICH == "synth":
        synth()
    elif WHICH == "compare":
        compare()
    else:
        raise SystemExit(f"unknown mode {WHICH}")
