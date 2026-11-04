"""把天梯上指定策略的对局 + 回放拉成「回放快照」同款目录，供 nn_build_bc / ladder_report 使用。

  python3 ladder_corpus.py OUT_DIR <名字前缀或策略ID>... [--loop 秒]
例：
  python3 ladder_corpus.py ../data/ladder_corpus claude-AlphaCarbon AgyCarbon-Argon-Godslayer Codex-Carbon CarbonEdge
  python3 ladder_corpus.py ../data/ladder_ours claude-opus55-cloud --loop 900    # 持续跟踪自家对局

目录结构：index.json（快照格式：id/game/order/scores/strategies/winner/steps/file）、leaderboard.json、
replays/<id>-<game>.json.gz。已下载的跳过；下载限速 ≤30 次/分钟（每次 2.1 s）。
需登录（cookie 由 ladder.py login 建立）。CORPUS_MIN_RATING=2150 只收高分版本。前缀按当前排行榜名字匹配（大小写敏感）。
"""
import gzip, json, os, sys, time

import ladder


MIN_RATING = float(os.environ.get("CORPUS_MIN_RATING", "0"))   # 只收评分 >= 该值的前缀匹配策略


def targets(keys, rows):
    out = {}
    for k in keys:
        for r in rows:
            if r["id"] == k or (r["name"].startswith(k) and (r.get("rating") or 0) >= MIN_RATING):
                out[r["id"]] = r["name"]
    return out


def sync(outdir, keys):
    os.makedirs(os.path.join(outdir, "replays"), exist_ok=True)
    ip = os.path.join(outdir, "index.json")
    index = {(g["id"], g["game"]): g for g in (json.load(open(ip)) if os.path.exists(ip) else [])}
    lb = ladder.request("/api/leaderboard")
    rows = lb["strategies"] + lb.get("extra", [])
    try:
        ladder.session()
        rows += [s for s in ladder.request("/api/mine")["strategies"] if s["id"] not in {r["id"] for r in rows}]
    except RuntimeError:
        pass
    json.dump(lb, open(os.path.join(outdir, "leaderboard.json"), "w"), ensure_ascii=False)
    tg = targets(keys, rows)
    print(f"{len(tg)} 个目标策略：", ", ".join(sorted(tg.values()))[:600], flush=True)
    new = 0
    for sid, name in tg.items():
      try:
        ms, off = [], 0
        while True:
            d = ladder.request(f"/api/matches?compact=1&limit=100&offset={off}&strategy={sid}")
            ms += d.get("matches", [])
            if d.get("next_offset") is None:
                break
            off = d["next_offset"]
            time.sleep(0.7)
        for m in ms:
            k = (m["id"], m["game"])
            if m.get("state") != "done" or not m.get("scores") or k in index:
                continue
            fn = f"replays/{m['id']}-{m['game']}.json.gz"
            path = os.path.join(outdir, fn)
            if not os.path.exists(path):
                st, data = ladder.request(f"/data/{m['id']}/{m['game']}", raw=True, timeout=120)
                time.sleep(2.1)
                if st != 200:
                    print(f"  {k}: HTTP {st}", flush=True)
                    continue
                if data[:2] != b"\x1f\x8b":
                    data = gzip.compress(data)
                open(path, "wb").write(data)
            try:
                steps = len(json.load(gzip.open(path))["steps"]) - 1
            except Exception as e:
                print(f"  {k}: 坏文件 {e}", flush=True)
                os.remove(path)
                continue
            # 按策略查询时 a 方 = 该策略
            index[k] = {"id": m["id"], "game": m["game"], "order": m["order"], "scores": m["scores"],
                        "winner": m.get("winner"), "steps": steps, "file": fn, "created": m.get("created"),
                        "strategies": {"a": {"id": sid, "name": m.get("name_a", name)},
                                       "b": {"id": "", "name": m.get("name_b", "?")}}}
            new += 1
            if new % 20 == 0:
                json.dump(list(index.values()), open(ip, "w"), ensure_ascii=False)
                print(f"  已入库 {len(index)} 局（本轮新增 {new}）", flush=True)
      except Exception as e:   # 单个策略失败不拖垮整轮
        print(f"  {name}: 失败 {e!r}，跳过（下轮再试）", flush=True)
        json.dump(list(index.values()), open(ip, "w"), ensure_ascii=False)
    json.dump(list(index.values()), open(ip, "w"), ensure_ascii=False)
    print(f"完成：共 {len(index)} 局，本轮新增 {new}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    loop = None
    if "--loop" in a:
        i = a.index("--loop"); loop = int(a[i + 1]); del a[i:i + 2]
    if len(a) < 2:
        sys.exit(__doc__)
    while True:
        try:
            sync(a[0], a[1:])
        except Exception as e:
            print("sync 失败：", repr(e), flush=True)
        if not loop:
            break
        time.sleep(loop)
