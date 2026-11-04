"""天梯战绩分析：python3 ladder_report.py <策略ID> <matches.json> [回放目录]

matches.json 来自 `ladder.py matches`；字段按回放快照 index.json 的格式（strategies/scores/winner/order/statuses）。
有回放目录时再做逐局拆解：终局现金、峰值采集员/种植员/树数、第 100/200 步现金、最后招募步。
输出只基于真实对局数据；少于 10 局时仅看方向。
"""
import collections, glob, gzip, json, math, os, sys


def side_of(m, sid):
    if "strategies" not in m and "name_a" in m:  # /api/matches?compact=1&strategy= 格式
        m["strategies"] = {"a": {"id": m.get("id_a", ""), "name": m["name_a"]},
                           "b": {"id": m.get("id_b", ""), "name": m["name_b"]}}
        if not m.get("id_a"):
            m["strategies"]["a"]["id"] = sid  # 按策略查询时 a 方即本策略
    for s, info in (m.get("strategies") or {}).items():
        if info.get("id") == sid:
            return s
    return None


def summarize(sid, ms):
    rows = []
    for m in ms:
        me = side_of(m, sid)
        if not me or not m.get("scores"):
            continue
        op = "b" if me == "a" else "a"
        sc = m["scores"]
        st = m.get("statuses") or {}
        rows.append(dict(id=m["id"], game=m.get("game"), me=me, op=op,
                         opp=m["strategies"][op].get("name", "?"),
                         diff=sc[me] - sc[op], mine=sc[me], theirs=sc[op],
                         win=(m.get("winner") == me), draw=(m.get("winner") in (None, "draw")),
                         status=f"{st.get(me, '?')}/{st.get(op, '?')}",
                         t=(m.get("action_timing") or {}).get(me, {}).get("total_seconds")))
    if not rows:
        print("无已结束对局")
        return rows
    n = len(rows)
    d = [r["diff"] for r in rows]
    mu = sum(d) / n
    se = (sum((x - mu) ** 2 for x in d) / max(1, n - 1)) ** 0.5 / math.sqrt(n)
    print(f"{n} 局  胜 {sum(r['win'] for r in rows)}  平均分差 {mu:+.0f} ± {se:.0f}  最差 {min(d):+.0f}")
    bad = [r for r in rows if not r["status"].startswith("DONE")]
    if bad:
        print(f"!! 己方非 DONE 状态 {len(bad)} 局：", [(r['id'][:8], r['status']) for r in bad])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["opp"]].append(r)
    print("\n对手                              局  胜   平均分差")
    for o, rs in sorted(by.items(), key=lambda kv: sum(r["diff"] for r in kv[1]) / len(kv[1])):
        print(f"{o[:32]:32s} {len(rs):3d} {sum(r['win'] for r in rs):3d} {sum(r['diff'] for r in rs) / len(rs):+9.0f}")
    return rows


def replay_stats(path, player):
    d = json.load(gzip.open(path))
    steps = d["steps"]
    out = dict(cash100=None, cash200=None, peak_col=0, peak_pla=0, peak_tree=0, last_recruit=None)
    for t, st in enumerate(steps):
        obs = st[0]["observation"]
        p = obs["players"][player]
        cash, workers, trees = p[0], p[2], (p[3] if len(p) > 3 else {})
        kinds = collections.Counter(w[2] for w in workers.values())
        out["peak_col"] = max(out["peak_col"], kinds.get("COLLECTOR", 0))
        out["peak_pla"] = max(out["peak_pla"], kinds.get("PLANTER", 0))
        out["peak_tree"] = max(out["peak_tree"], len(trees))
        if t == 100: out["cash100"] = cash
        if t == 200: out["cash200"] = cash
        act = st[player].get("action") or {}
        if any(v in ("RECCOLLECTOR", "RECPLANTER") for v in act.values()):
            out["last_recruit"] = t
    out["final"] = d["rewards"][player]
    return out


def main():
    sid, mfile = sys.argv[1], sys.argv[2]
    rdir = sys.argv[3] if len(sys.argv) > 3 else None
    ms = json.load(open(mfile))
    rows = summarize(sid, ms)
    if not (rdir and rows):
        return
    print("\n逐局（按分差升序）  我/敌：现金@100 现金@200 终局 | 峰值采/种/树 | 最后招募步")
    for r in sorted(rows, key=lambda r: r["diff"]):
        f = os.path.join(rdir, f"{r['id']}-{r['game']}.json.gz")
        m = next(x for x in ms if x["id"] == r["id"] and x.get("game") == r["game"])
        order = m.get("order") or ["a", "b"]
        if not os.path.exists(f):
            continue
        pm, po = order.index(r["me"]), order.index(r["op"])
        a, b = replay_stats(f, pm), replay_stats(f, po)
        fmt = lambda s: (f"{s['cash100'] or 0:6.0f} {s['cash200'] or 0:6.0f} {s['final']:6.0f} | "
                         f"{s['peak_col']:2d}/{s['peak_pla']:2d}/{s['peak_tree']:2d} | {s['last_recruit']}")
        print(f"{r['diff']:+7.0f} vs {r['opp'][:20]:20s} {r['id'][:8]}-{r['game']}\n   我 {fmt(a)}\n   敌 {fmt(b)}")


if __name__ == "__main__":
    main()
