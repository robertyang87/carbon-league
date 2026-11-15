#!/usr/bin/env python3
"""新面孔巡检 20:56 趟：117 端拉全量清单，与仓内 main data/<label>_matches.json
比对 id 集，有变化才 put_file 并 get_file 读回核对；同时数各 label 语料
data/<label>_corpus 的文件数算缺口。it570 / it260 冻结件只核实不覆盖。
名字守卫：拉回清单须全含目标名，否则跳过不提交（09:56 m9xj 错提交教训）。"""
import base64, json, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST = "https://117.72.210.134"
# (label, strategy_id, display_name, frozen_verify_only)
LABELS = [
    ("m3s3z", "f9fdbce38def1a78eb9c77f8711a1103", "claude-AlphaCarbon-Master3s3z", False),
    ("m9xj570", "28c19bba51a260bbae29ea5fba59951c", "RyanX-AIPowercore-M9XJ-it570", True),
    ("hakimi3pro", "aff6e57c5eba8857d5546783c7d39a46", "哈基米 v3pro", False),
    ("k4v3", "9d596068f39e0b8cca317c3bb27f8ae5", "stephen-K4TacSearch-v3", False),
    ("fable6", "5b453cb4f7c49439673deda697d46a9b", "Fable5.2-Max-V6", False),
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d", "RyanX-AIPowercore-Lite-V8", False),
    ("krypton", "0f2c23daf04aaf650576575d2482b940", "Krypton-Cloud500", False),
    ("fable", "2b4b44fd148fa7fdaa53bb0d110ae705", "Fable5.2-Max-V5", False),
    ("k4v2", "aaa02959b15d57e4e28f80d665ae3482", "stephen-K4TacSearch-v2", False),
    ("krypton-search256", "2c57e7cd01dee46bef98643398ab4ca3", "Krypton-Search256-v1", False),
    ("krypton-prosearch", "fede68b79a2965889eeae0e226601139", "Krypton-ProSearch-v1", False),
    ("krypton-search256-v2", "484814899c333f8149e984eacdaf5275", "Krypton-Search256-v2", False),
    ("krypton-cloudsearch", "6e830c045157f572ad460c4a8a37b822", "Krypton-CloudSearch-v1", False),
    ("m9xj", "e43b20c5d8d644b4145cb1e7ff198201", "RyanX-AIPowercore-M9XJ-it260", True),
    ("hakimi31", "29844f8f0c9b14b3d37902a551d26a1e", "哈基米 v3.1", True),
    ("m3s3", "f47e13ba624a85c5c2dc1b40caecc193", "claude-AlphaCarbon-Master3s3", True),
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

for label, sid, dname, frozen in LABELS:
    try:
        fresh = fetch117(sid)
    except Exception as e:
        print(f"{label}: FETCH117 FAIL {e}", flush=True)
        continue
    bad = [m for m in fresh if dname not in (m.get("name_a"), m.get("name_b"))]
    if bad:
        print(f"{label}: NAME-GUARD FAIL {len(bad)}/{len(fresh)} 条不含 {dname}，跳过不提交", flush=True)
        continue
    old, sha = repo_matches(label)
    if old is None:
        print(f"{label}: 117={len(fresh)} 仓内无文件", flush=True)
        continue
    old_ids = {(m.get("id"), m.get("game")) for m in old}
    new_ids = {(m.get("id"), m.get("game")) for m in fresh}
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)
    cc = corpus_count(label)
    print(f"{label}: 117={len(fresh)} 仓内={len(old)} 新增={added} 删减={removed} sha={sha[:12]} corpus={cc}", flush=True)
    if frozen:
        print(f"  -> 冻结/终局件只核实不覆盖（仓内 {len(old)} 局）", flush=True)
        continue
    if added == 0 and removed == 0:
        print("  -> 无变化，不提交", flush=True)
        continue
    if removed:
        print("  -> 注意：出现删减，暂不提交，留待人工核对", flush=True)
        continue
    body = json.dumps(fresh, indent=1, ensure_ascii=False).encode()
    r = gh_api.put_file("robertyang87", "carbon-league", f"data/{label}_matches.json",
                        body, f"newface-watch 2056: {label} matches {len(old)}->{len(fresh)}", sha=sha)
    back, sha2 = repo_matches(label)
    print(f"  -> 已提交 commit={r.get('commit', {}).get('sha', '?')[:12] if isinstance(r, dict) else r} 读回={len(back)} sha={sha2[:12]}", flush=True)
