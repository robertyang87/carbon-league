#!/usr/bin/env python3
"""newface-watch 18:26 run: refresh data/<label>_matches.json from the 117
endpoint (curl, paginated), compare id sets with repo main, put_file only on
change, get_file read-back verify. Also counts data/<label>_corpus files via
the git tree for the corpus-gap check."""
import json, subprocess, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST = "117.72.210.134"
LABELS = {
    "m3s3z": "f9fdbce38def1a78eb9c77f8711a1103",
    "hakimi3pro": "aff6e57c5eba8857d5546783c7d39a46",
    "k4v3": "9d596068f39e0b8cca317c3bb27f8ae5",
    "fable6": "5b453cb4f7c49439673deda697d46a9b",
    "lite-v8": "3a4d267d9fcea50794f48d38d3e1360d",
    "krypton": "0f2c23daf04aaf650576575d2482b940",
    "k4v2": "aaa02959b15d57e4e28f80d665ae3482",
    "fable": "2b4b44fd148fa7fdaa53bb0d110ae705",
    "m9xj": "e43b20c5d8d644b4145cb1e7ff198201",  # it260 的正确 ID；28cd4ac93... 是 it320，曾误用致错提交 bad1af1796df（已恢复 1d57e6711c46）
    "krypton-search256": "2c57e7cd01dee46bef98643398ab4ca3",  # 12:56 刚满 100 局、第 16 名、上升快，保险建档
    "krypton-prosearch": "fede68b79a2965889eeae0e226601139",  # 16:26 刚满 100 局、第 15 名、上升快，保险建档（沿 12:56 Search256-v1 先例）
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
    # guard: every match must involve the label's own strategy name
    return uniq


EXPECTED_NAME = {
    "m3s3z": "Master3s3z", "hakimi3pro": "哈基米 v3pro",
    "k4v3": "K4TacSearch-v3", "fable6": "Fable5.2-Max-V6",
    "lite-v8": "Lite-V8", "krypton": "Krypton-Cloud500",
    "k4v2": "K4TacSearch-v2", "fable": "Fable5.2-Max-V5",
    "m9xj": "M9XJ-it260",
    "krypton-search256": "Krypton-Search256-v1",
    "krypton-prosearch": "Krypton-ProSearch-v1",
}

# corpus counts from one recursive tree read
tree = gh_api.req("GET", "/repos/robertyang87/carbon-league/git/trees/main?recursive=1")
corpus = {}
for t in tree.get("tree", []):
    p = t["path"]
    if p.startswith("data/") and p.endswith("_corpus/"):
        pass
for lab in list(LABELS) + ["m9xj570", "hakimi31", "m3s3"]:
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
    try:
        r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
        cur = json.loads(base64.b64decode(r["content"]))
        cur_sha = r["sha"]
    except Exception:
        cur, cur_sha = [], None
        print(f"{lab}: no existing file in repo; will create")
    cur_ids = {(m["id"], m.get("game")) for m in cur}
    new_ids = {(m["id"], m.get("game")) for m in fresh}
    added = new_ids - cur_ids
    removed = cur_ids - new_ids
    if not added and not removed:
        print(f"{lab}: no change ({len(cur)}), sha {r['sha'][:12]}")
        results[lab] = {"n": len(cur), "changed": False, "sha": (cur_sha or "")[:12]}
        continue
    body = json.dumps(fresh, indent=1, ensure_ascii=False).encode()
    put = gh_api.put_file("robertyang87", "carbon-league", f"data/{lab}_matches.json",
                          body, f"newface-watch 1826: {lab} {len(cur)}->{len(fresh)}",
                          sha=cur_sha, branch="main")
    rb = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    rb_n = len(json.loads(base64.b64decode(rb["content"])))
    print(f"{lab}: {len(cur)}->{len(fresh)} (+{len(added)} -{len(removed)}) "
          f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={rb_n} sha={rb['sha'][:12]}")
    results[lab] = {"n": len(fresh), "changed": True, "added": len(added),
                    "removed": len(removed),
                    "commit": put.get("commit", {}).get("sha", "?")[:12],
                    "readback": rb_n, "sha": rb["sha"][:12]}

# frozen / vanished files: sha-only verification
for lab in ["m9xj570", "hakimi31", "m3s3"]:
    r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    n = len(json.loads(base64.b64decode(r["content"])))
    print(f"{lab}: verify-only n={n} sha={r['sha'][:12]}")
    results[lab] = {"n": n, "changed": False, "sha": r["sha"][:12], "verify_only": True}

json.dump(results, open("/tmp/refresh_1826_results.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
