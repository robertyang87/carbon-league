"""probe_read STRATEGY_ID NAME OUT_DIR -- pull a probe's ladder games and report ONLY those against the M family
(RyanX-AIPowercore-*) and the Master family (claude-AlphaCarbon-Master*): margin, trees at 50 / 100, planted / seized
in steps 1-100 (ownership changes), cargo still aboard at the end; per-family means."""
import gzip, json, os, subprocess, sys, statistics as S
sid, name, out = sys.argv[1:4]
env = dict(os.environ, LADDER_BASE=os.environ.get("LADDER_BASE", "https://117.72.210.134"))
os.makedirs(os.path.join(out, "rp"), exist_ok=True)
subprocess.run([sys.executable, "ladder.py", "matches", sid, os.path.join(out, "m.json")], env=env, capture_output=True)
ms = json.load(open(os.path.join(out, "m.json")))
keep = [m for m in ms if any(k in (m["name_a"] + m["name_b"]) for k in ("RyanX-AIPowercore", "claude-AlphaCarbon-Master"))]
json.dump(keep, open(os.path.join(out, "keep.json"), "w"))
subprocess.run([sys.executable, "ladder.py", "replays", os.path.join(out, "keep.json"), os.path.join(out, "rp")], env=env, capture_output=True)
fam = {"M": [], "Master": []}
for m in keep:
    f = os.path.join(out, "rp", f"{m['id']}-{m.get('game', 1)}.json.gz")
    if not os.path.exists(f):
        continue
    st = json.load(gzip.open(f))["steps"]
    who = "a" if m["name_a"] == name else "b"; op = m["name_b"] if who == "a" else m["name_a"]
    me = m["order"].index(who)
    P = lambda t, p: st[t][0]["observation"]["players"][p]
    pl = {0: [0, 0], 1: [0, 0]}
    for t in range(1, 101):
        for p in (0, 1):
            for tid in P(t, p)[3]:
                if tid not in P(t - 1, p)[3]:
                    pl[p][1 if tid in P(t - 1, 1 - p)[3] else 0] += 1
    T = len(st) - 1
    seg = []      # per 100-step segment: car-steps / planter-steps (ours, theirs)
    for a0, a1 in ((1, 100), (101, 200), (201, T)):
        cs = [0, 0]; ps = [0, 0]
        for t in range(a0, a1 + 1):
            for q, p in ((0, me), (1, 1 - me)):
                for w in P(t, p)[2].values():
                    (cs if w[2] == "COLLECTOR" else ps)[q] += 1
        seg.append((cs, ps))
    r = dict(op=op, margin=m["scores"][who] - m["scores"]["b" if who == "a" else "a"], t50=(len(P(50, me)[3]), len(P(50, 1 - me)[3])),
             t100=(len(P(100, me)[3]), len(P(100, 1 - me)[3])), pl=(pl[me][0], pl[me][1], pl[1 - me][0], pl[1 - me][1]),
             strand=(sum(float(w[1]) for w in P(T, me)[2].values()), sum(float(w[1]) for w in P(T, 1 - me)[2].values())))
    r["seg"] = seg
    fam["M" if "RyanX" in op else "Master2s" if op.endswith("Master2s") else "Master"].append(r)
    print(f"{op:30s} {r['margin']:+7.0f}  t50 {r['t50'][0]}/{r['t50'][1]}  t100 {r['t100'][0]}/{r['t100'][1]}  planted/seized us {r['pl'][0]}/{r['pl'][1]} them {r['pl'][2]}/{r['pl'][3]}  cargo left {r['strand'][0]:.0f}/{r['strand'][1]:.0f}")
for k, v in fam.items():
    if v:
        mm = lambda f: S.mean(f(x) for x in v)
        print(f"== vs {k} family, {len(v)} games: wins {sum(x['margin'] > 0 for x in v)}, margin {mm(lambda x: x['margin']):+.0f}, t50 {mm(lambda x: x['t50'][0]):.1f}/{mm(lambda x: x['t50'][1]):.1f}, "
              f"t100 {mm(lambda x: x['t100'][0]):.1f}/{mm(lambda x: x['t100'][1]):.1f}, planted/seized us {mm(lambda x: x['pl'][0]):.1f}/{mm(lambda x: x['pl'][1]):.1f} them {mm(lambda x: x['pl'][2]):.1f}/{mm(lambda x: x['pl'][3]):.1f}, cargo left {mm(lambda x: x['strand'][0]):.0f}/{mm(lambda x: x['strand'][1]):.0f}")
        print("     car-steps us/them by segment " + "  ".join(f"{mm(lambda x, i=i: x['seg'][i][0][0]):.0f}/{mm(lambda x, i=i: x['seg'][i][0][1]):.0f}" for i in range(3))
              + " | planter-steps " + "  ".join(f"{mm(lambda x, i=i: x['seg'][i][1][0]):.0f}/{mm(lambda x, i=i: x['seg'][i][1][1]):.0f}" for i in range(3)))
