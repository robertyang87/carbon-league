#!/usr/bin/env python3
"""新面孔巡检 06:56 趟：117 端拉全量清单，与仓内 main data/<label>_matches.json
比对 id 集，有变化才 put_file 并 get_file 读回核对；同时数各 label 语料
data/<label>_corpus 的文件数算缺口。it570 冻结件只核实不覆盖。"""
import base64, json, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST = "https://117.72.210.134"
LABELS = [
    ("m3s3z", "f9fdbce38def1a78eb9c77f8711a1103"),
    ("m9xj570", "28c19bba51a260bbae29ea5fba59951c"),
    ("hakimi3pro", "aff6e57c5eba8857d5546783c7d39a46"),
    ("k4v3", "9d596068f39e0b8cca317c3bb27f8ae5"),
    ("fable6", "5b453cb4f7c49439673deda697d46a9b"),
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d"),
    ("krypton", "0f2c23daf04aaf650576575d2482b940"),
    ("fable", "2b4b44fd148fa7fdaa53bb0d110ae705"),
    ("k4v2", "aaa02959b15d57e4e28f80d665ae3482"),
]

def fetch117(sid):
    got, off = [], 0
    while True:
        url = f"{HOST}/api/matches?compact=1&limit=100&offset={off}&strategy={sid}"
        out = subprocess.run(["curl", "-skL", url], capture_output=True, text=True, timeout=60)
        d = json.loads(out.stdout)
        got += d.get("matches", [])
        if d.get("next_offset") is None:
            break
        off = d["next_offset"]
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq

def repo_matches(label):
    f = gh_api.get_file("robertyang87", "carbon-league", f"data/{label}_matches.json")
    if f is None:
        return None, None
    return json.loads(base64.b64decode(f["content"]).decode()), f["sha"]

def corpus_count(label):
    try:
        r = gh_api.req("GET", f"/repos/robertyang87/carbon-league/contents/data/{label}_corpus?ref=main")
        if isinstance(r, list):
            return len([e for e in r if e.get("type") == "file"])
        return f"ERR:{str(r)[:80]}"
    except Exception as e:
        return f"ERR:{e}"

for label, sid in LABELS:
    try:
        fresh = fetch117(sid)
    except Exception as e:
        print(f"{label}: FETCH117 FAIL {e}")
        continue
    wins = sum(1 for m in fresh if m.get("winner") == ("a" if m.get("name_a") and True else None))
    # winner is 'a'/'b' relative to name_a/name_b; count wins for the strategy by name
    # recompute properly below
    old, sha = repo_matches(label)
    if old is None:
        print(f"{label}: 117={len(fresh)} 仓内无文件")
        continue
    old_ids = {(m.get("id"), m.get("game")) for m in old}
    new_ids = {(m.get("id"), m.get("game")) for m in fresh}
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)
    # wins for this strategy: winner side name equals strategy name? use name fields
    w = 0
    for m in fresh:
        side = m.get("winner")
        if side in ("a", "b"):
            nm = m.get("name_a") if side == "a" else m.get("name_b")
            # strategy name unknown here; wins counted by caller via board; just record total
    print(f"{label}: 117={len(fresh)} 仓内={len(old)} 新增={added} 删减={removed} sha={sha[:12]} corpus={corpus_count(label)}")
    if label == "m9xj570":
        print(f"  -> 冻结件只核实不覆盖（仓内 {len(old)} 局，预期 89）")
        continue
    if added == 0 and removed == 0:
        print("  -> 无变化，不提交")
        continue
    if removed:
        print("  -> 注意：出现删减，暂不提交，留待人工核对")
        continue
    body = json.dumps(fresh, indent=1, ensure_ascii=False).encode()
    r = gh_api.put_file("robertyang87", "carbon-league", f"data/{label}_matches.json",
                        body, f"newface-watch 0656: {label} matches {len(old)}->{len(fresh)}", sha=sha)
    back, sha2 = repo_matches(label)
    print(f"  -> 已提交 commit={r.get('commit', {}).get('sha', '?')[:12] if isinstance(r, dict) else r} 读回={len(back)} sha={sha2[:12]}")
