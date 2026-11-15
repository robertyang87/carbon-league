#!/usr/bin/env python3
"""newface-watch 13:56 run 2026-11-15: refresh data/<label>_matches.json from
the 117 endpoint (curl, paginated), compare id sets with repo main, put_file
only on change, get_file read-back verify. Also live 117 counts for
it570/it320 (no repo write) and corpus counts from one recursive tree read."""
import json, subprocess, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST = "117.72.210.134"
LABELS = {
    "master4s": "6389c346461bce295badcf52a1dd74c6",
    "m9xj630": "b3fa633052e17329f34054bab0090128",
    "xiaofeiyu5": "2856c37287900ee3cba8eca53b827da9",
    "m9xj600": "b1cde1fda59f9bab81e56f40de92302d",
    "m3s3z": "f9fdbce38def1a78eb9c77f8711a1103",
    "hakimi3pro": "aff6e57c5eba8857d5546783c7d39a46",
    "k4v3": "9d596068f39e0b8cca317c3bb27f8ae5",
    "fable6": "5b453cb4f7c49439673deda697d46a9b",
    "lite-v8": "3a4d267d9fcea50794f48d38d3e1360d",
    "krypton": "0f2c23daf04aaf650576575d2482b940",
    "k4v2": "aaa02959b15d57e4e28f80d665ae3482",
    "fable": "2b4b44fd148fa7fdaa53bb0d110ae705",
    "m9xj": "e43b20c5d8d644b4145cb1e7ff198201",
    "krypton-m128": "96db8cfd61a73a534807cc404d046e31",
    "krypton-cloudsearch": "6e830c045157f572ad460c4a8a37b822",
    "krypton-csplant": "adaf275d51a177b7a9cf8224ab66c467",
}
LIVE_ONLY = {
    "it570": "28c19bba51a260bbae29ea5fba59951c",
    "it320": "28cd4ac93ab020aa6c1abd821675339d",
    "krypton-search256": "2c57e7cd01dee46bef98643398ab4ca3",
    "krypton-prosearch": "fede68b79a2965889eeae0e226601139",
    "krypton-search256-v2": "484814899c333f8149e984eacdaf5275",
}
EXPECTED_NAME = {
    "master4s": "claude-AlphaCarbon-Master4s",
    "m9xj630": "RyanX-AIPowercore-M9XJ-630",
    "xiaofeiyu5": "小肥鱼v5",
    "m9xj600": "RyanX-AIPowercore-M9XJ-600",
    "m3s3z": "claude-AlphaCarbon-Master3s3z",
    "hakimi3pro": "哈基米 v3pro",
    "k4v3": "K4TacSearch-v3",
    "fable6": "Fable5.2-Max-V6",
    "lite-v8": "Lite-V8",
    "krypton": "Krypton-Cloud500",
    "k4v2": "K4TacSearch-v2",
    "fable": "Fable5.2-Max-V5",
    "m9xj": "M9XJ-it260",
    "krypton-m128": "Krypton-M128Search-v1",
    "krypton-cloudsearch": "Krypton-CloudSearch-v1",
    "krypton-csplant": "Krypton-CS-Plant-v1",
    "it570": "M9XJ-it570",
    "it320": "M9XJ-it320",
    "krypton-search256": "Krypton-Search256-v1",
    "krypton-prosearch": "Krypton-ProSearch-v1",
    "krypton-search256-v2": "Krypton-Search256-v2",
}

def pull(sid):
    got, off = [], 0
    while True:
        url = f"https://{HOST}/api/matches?compact=1&limit=100&offset={off}&strategy={sid}"
        out = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"curl fail {url}: {out.stderr[:200]}")
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

# corpus counts from one recursive tree read
tree = gh_api.req("GET", "/repos/robertyang87/carbon-league/git/trees/main?recursive=1")
corpus = {}
for lab in list(LABELS) + ["m9xj570", "hakimi31", "m3s3", "krypton-search256",
                           "krypton-prosearch", "krypton-search256-v2"]:
    n = sum(1 for t in tree.get("tree", [])
            if t["path"].startswith(f"data/{lab}_corpus/") and t["type"] == "blob")
    corpus[lab] = n
print("corpus:", json.dumps(corpus, ensure_ascii=False))

results = {}
for lab, sid in LABELS.items():
    try:
        fresh = pull(sid)
    except Exception as e:
        results[lab] = {"error": str(e)}
        print(lab, "PULL-ERR", e)
        continue
    want = EXPECTED_NAME[lab]
    if not all(want in (m.get("name_a", "") + m.get("name_b", "")) for m in fresh):
        print(f"{lab}: NAME-GUARD FAIL — pulled matches do not all involve {want}; SKIPPING put")
        results[lab] = {"error": f"name-guard fail for {want}", "n_pulled": len(fresh)}
        continue
    r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    cur = json.loads(base64.b64decode(r["content"]))
    cur_ids = {(m["id"], m.get("game")) for m in cur}
    new_ids = {(m["id"], m.get("game")) for m in fresh}
    added = new_ids - cur_ids
    removed = cur_ids - new_ids
    if not added and not removed:
        print(f"{lab}: no change ({len(cur)}), sha {r['sha'][:12]}")
        results[lab] = {"n": len(cur), "changed": False, "sha": r["sha"][:12]}
        continue
    body = json.dumps(fresh, indent=1, ensure_ascii=False).encode()
    put = gh_api.put_file("robertyang87", "carbon-league", f"data/{lab}_matches.json",
                          body, f"newface-watch 1356: {lab} {len(cur)}->{len(fresh)}",
                          sha=r["sha"], branch="main")
    rb = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    rb_n = len(json.loads(base64.b64decode(rb["content"])))
    print(f"{lab}: {len(cur)}->{len(fresh)} (+{len(added)} -{len(removed)}) "
          f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={rb_n} sha={rb['sha'][:12]}")
    results[lab] = {"n": len(fresh), "changed": True, "added": len(added),
                    "removed": len(removed),
                    "commit": put.get("commit", {}).get("sha", "?")[:12],
                    "readback": rb_n, "sha": rb["sha"][:12]}

# live-only 117 counts (no repo write)
for lab, sid in LIVE_ONLY.items():
    try:
        fresh = pull(sid)
    except Exception as e:
        results[lab] = {"error": str(e)}
        print(lab, "PULL-ERR", e)
        continue
    wins = sum(1 for m in fresh
               if (m.get("winner") == "a" and EXPECTED_NAME[lab] in m.get("name_a", ""))
               or (m.get("winner") == "b" and EXPECTED_NAME[lab] in m.get("name_b", "")))
    print(f"{lab}: live 117 n={len(fresh)} wins={wins}")
    results[lab] = {"n": len(fresh), "wins": wins, "live_only": True}

# frozen / vanished files: sha-only verification
for lab in ["m9xj570", "hakimi31", "m3s3", "krypton-search256",
            "krypton-prosearch", "krypton-search256-v2"]:
    r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    n = len(json.loads(base64.b64decode(r["content"])))
    print(f"{lab}: verify-only n={n} sha={r['sha'][:12]}")
    results[lab] = {"n": n, "changed": False, "sha": r["sha"][:12], "verify_only": True}

json.dump(results, open("/tmp/refresh_1356_1115_results.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
