"""ladder_ledger_cmp NAME_A NAME_B [--min N] -- side-by-side ladder ledger of two of OUR strategies from their ladder
replays (fetches match lists + replays via ladder.py; needs the bridge env: LADDER_BASE/LADDER_HOST/BRIDGE_TARGET).

Per game, from the replay alone (no local sim):  strand (cargo on our cars at step 299), our trees at 150/200/250,
our / opp tree-steps, our / opp trees PLANTED, opp seized, our / opp "tree net" (cash - deposits + 30*recruits:
absorption income minus plant/seize costs), cargo seized by the enemy (attribution: an enemy car's cargo grew by
>= 90% of the vanished car's load), own merges, and the result.  Prints means +- s.e. for both and the difference.
Use it at ~30 ladder games to see how much of a paired-exam gain shows up on the ladder (对方种树数 included)."""
import gzip, json, os, statistics as st, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)
import ladder  # noqa: E402

OUT = os.environ.get("LADDER_CMP_OUT", os.path.join(HERE, "ladder_cmp"))
S = 15


def xy(i): return (int(i) // S, int(i) % S)


def dist(a, b):
    dr = abs(a[0] - b[0]); dc = abs(a[1] - b[1]); return min(dr, S - dr) + min(dc, S - dc)


def fetch(name):
    """match rows (compact) + replays for the ladder strategy called NAME -> (rows, replay dir)"""
    ladder.session()
    b = ladder.request("/api/leaderboard")
    rows = [r for v in b.values() if isinstance(v, list) for r in v]
    sid = [r["id"] for r in rows if r["name"] == name]
    if not sid:
        sys.exit(f"{name}: not on the leaderboard")
    sid = sid[0]
    os.makedirs(OUT, exist_ok=True)
    mf = os.path.join(OUT, f"m_{sid}.json"); rd = os.path.join(OUT, f"rep_{sid}")
    ladder.cmd_matches(sid, mf)              # follows next_offset (the API caps a page at 25)
    ladder.cmd_replays(mf, rd)               # <= 30 downloads / minute
    return json.load(open(mf)), rd


def ledger(d, p):
    """one side's ledger from a replay dict"""
    stp = d["steps"]; q = 1 - p; prev = None
    L = dict(strand=0.0, t150=0, t200=0, t250=0, ts=0, ots=0, planted=0, oplanted=0, seized=0, oseized=0,
             dep=0.0, rec=0, odep=0.0, orec=0, lost_en=0.0, lost_self=0.0, late30_en=0.0, took=0.0)
    for t, s in enumerate(stp):
        o = s[0]["observation"]; P = o["players"][p]; O = o["players"][q]
        B = {xy(b) for b in P[1].values()}; OB = {xy(b) for b in O[1].values()}
        L["ts"] += len(P[3]); L["ots"] += len(O[3])
        if t in (150, 200, 250): L[f"t{t}"] = len(P[3])
        if prev is not None:
            pP, pO = prev
            for tid in P[3]:
                if tid not in pP[3]: L["planted" if tid.startswith(f"player-{p}-") else "seized"] += 1
            for tid in O[3]:
                if tid not in pO[3]: L["oplanted" if tid.startswith(f"player-{q}-") else "oseized"] += 1
            L["rec"] += len(set(P[2]) - set(pP[2])); L["orec"] += len(set(O[2]) - set(pO[2]))
            for u, w in P[2].items():
                if u in pP[2] and w[2] == "COLLECTOR" and pP[2][u][1] > 5 and w[1] < 0.5 and xy(w[0]) in B: L["dep"] += pP[2][u][1]
            for u, w in O[2].items():
                if u in pO[2] and w[2] == "COLLECTOR" and pO[2][u][1] > 5 and w[1] < 0.5 and xy(w[0]) in OB: L["odep"] += pO[2][u][1]
            egain = [O[2][u][1] - pO[2][u][1] for u in O[2] if u in pO[2]]
            ogain = [P[2][u][1] - pP[2][u][1] for u in P[2] if u in pP[2]]
            for u, w in pP[2].items():
                if u in P[2] or w[2] != "COLLECTOR" or w[1] <= 0: continue
                c = w[1]
                if any(g >= 0.9 * c for g in egain):
                    L["lost_en"] += c
                    if t > 270: L["late30_en"] += c
                elif any(g >= 0.9 * c for g in ogain): L["lost_self"] += c
            for u, w in pO[2].items():
                if u not in O[2] and w[2] == "COLLECTOR" and w[1] > 0 and any(g >= 0.9 * w[1] for g in ogain): L["took"] += w[1]
        prev = (P, O)
    P = stp[-1][0]["observation"]["players"][p]; O = stp[-1][0]["observation"]["players"][q]
    L["strand"] = sum(w[1] for w in P[2].values())
    L["tree_net"] = P[0] - (L["dep"] - 30 * L["rec"])        # absorption income minus plant / seize costs
    L["otree_net"] = O[0] - (L["odep"] - 30 * L["orec"])
    L["margin"] = d["rewards"][p] - d["rewards"][q]
    return L


def collect(name, rows, rd):
    out = []
    for m in rows:
        f = os.path.join(rd, f"{m['id']}-{m.get('game', 1)}.json.gz")
        if not os.path.exists(f) or m.get("winner") is None: continue
        d = json.load(gzip.open(f))
        order = m.get("order") or ["a", "b"]
        names = [m["name_" + order[0]], m["name_" + order[1]]]
        p = names.index(name) if name in names else -1
        if p < 0: continue
        L = ledger(d, p); L["opp"] = names[1 - p]; out.append(L)
    return out


KEYS = [("margin", "分差"), ("strand", "剩货 strand"), ("late30_en", "最后30步被敌截"), ("lost_en", "被敌截(全局)"), ("lost_self", "己方合并"),
        ("took", "截到对方"), ("t150", "我方树 t150"), ("t200", "我方树 t200"), ("t250", "我方树 t250"), ("ts", "我方树·步"), ("ots", "对方树·步"),
        ("planted", "我方种树数"), ("oplanted", "对方种树数"), ("seized", "我方抢树"), ("oseized", "对方抢树"),
        ("tree_net", "我方树净收入"), ("otree_net", "对方树净收入"), ("dep", "我方入库"), ("odep", "对方入库")]


def table(A, B, na, nb, title):
    print(f"\n## {title}: {na} n={len(A)}  vs  {nb} n={len(B)}")
    print(f"{'指标':14s} {na[-10:]:>16s} {nb[-10:]:>16s} {'差(A-B)':>10s}")
    for k, lab in KEYS:
        a = [r[k] for r in A]; b = [r[k] for r in B]
        if not a or not b: continue
        sa = st.stdev(a) / len(a) ** .5 if len(a) > 1 else 0; sb = st.stdev(b) / len(b) ** .5 if len(b) > 1 else 0
        print(f"{lab:14s} {st.mean(a):9.0f} ±{sa:4.0f} {st.mean(b):9.0f} ±{sb:4.0f} {st.mean(a)-st.mean(b):+10.0f}")
    wa = sum(r["margin"] > 0 for r in A); wb = sum(r["margin"] > 0 for r in B)
    print(f"胜局 {wa}/{len(A)} vs {wb}/{len(B)} | 剩货>分差的输局: {sum(1 for r in A if r['margin'] < 0 and r['strand'] > -r['margin'])} vs {sum(1 for r in B if r['margin'] < 0 and r['strand'] > -r['margin'])}")


def main():
    a = sys.argv[1:]; mn = 0
    if "--min" in a:
        i = a.index("--min"); mn = int(a[i + 1]); del a[i:i + 2]
    na, nb = a[0], a[1]
    ra, da = fetch(na); rb, db = fetch(nb)
    A, B = collect(na, ra, da), collect(nb, rb, db)
    if mn and (len(A) < mn or len(B) < mn):
        print(f"only {len(A)} / {len(B)} games (need {mn}); exit 2"); sys.exit(2)
    table(A, B, na, nb, "全部对局")
    top = ("Master", "RyanX-AIPowercore-M1", "1C-Zero", "无始", "cloud-v1")
    table([r for r in A if any(t in r["opp"] for t in top)], [r for r in B if any(t in r["opp"] for t in top)], na, nb, "对头部 (Master/M1/1C/无始/我方)")
    # same-opponent subset: fairest ladder comparison short of paired maps
    common = {r["opp"] for r in A} & {r["opp"] for r in B}
    table([r for r in A if r["opp"] in common], [r for r in B if r["opp"] in common], na, nb, f"共同对手 ({len(common)} 家)")


if __name__ == "__main__":
    main()
