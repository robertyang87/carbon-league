#!/usr/bin/env python3
"""newface-watch 11:26 run follow-up: refresh the fast movers + remaining
Krypton labels, live-count checks for it570/it320, sha-verify for the three
vanished Krypton members. Same put/get-back pattern as refresh_1126.py."""
import json, subprocess, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST = "117.72.210.134"
LABELS = {
    "m9xj630": ("b3fa633052e17329f34054bab0090128", "RyanX-AIPowercore-M9XJ-630"),
    "m9xj600": ("b1cde1fda59f9bab81e56f40de92302d", "RyanX-AIPowercore-M9XJ-600"),
    "xiaofeiyu5": ("2856c37287900ee3cba8eca53b827da9", "小肥鱼v5"),
    "krypton-cloudsearch": ("6e830c045157f572ad460c4a8a37b822", "Krypton-CloudSearch-v1"),
    "krypton-m128": ("96db8cfd61a73a534807cc404d046e31", "Krypton-M128Search-v1"),
    "krypton-csplant": ("adaf275d51a177b7a9cf8224ab66c467", "Krypton-CS-Plant-v1"),
}
LIVE_ONLY = {
    "it570": ("28c19bba51a260bbae29ea5fba59951c", "RyanX-AIPowercore-M9XJ-it570"),
    "it320": ("28cd4ac93ab020aa6c1abd821675339d", "RyanX-AIPowercore-M9XJ-it320"),
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

tree = gh_api.req("GET", "/repos/robertyang87/carbon-league/git/trees/main?recursive=1")
corpus = {}
for lab in list(LABELS) + ["krypton-search256", "krypton-prosearch", "krypton-search256-v2"]:
    corpus[lab] = sum(1 for t in tree.get("tree", [])
                      if t["path"].startswith(f"data/{lab}_corpus/") and t["type"] == "blob")
print("corpus:", json.dumps(corpus, ensure_ascii=False))

results = {}
for lab, (sid, want) in LABELS.items():
    fresh = pull(sid)
    json.dump(fresh, open(f"/tmp/{lab}_117.json", "w"), indent=1, ensure_ascii=False)
    if not all(want in (m.get("name_a", "") + m.get("name_b", "")) for m in fresh):
        print(f"{lab}: NAME-GUARD FAIL; SKIPPING put")
        results[lab] = {"error": "name-guard fail", "n_pulled": len(fresh)}
        continue
    r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    cur = json.loads(base64.b64decode(r["content"]))
    cur_ids = {(m["id"], m.get("game")) for m in cur}
    new_ids = {(m["id"], m.get("game")) for m in fresh}
    added, removed = new_ids - cur_ids, cur_ids - new_ids
    if not added and not removed:
        print(f"{lab}: no change ({len(cur)}), sha {r['sha'][:12]}")
        results[lab] = {"n": len(cur), "changed": False, "sha": r["sha"][:12]}
        continue
    body = json.dumps(fresh, indent=1, ensure_ascii=False).encode()
    put = gh_api.put_file("robertyang87", "carbon-league", f"data/{lab}_matches.json",
                          body, f"newface-watch 1126: {lab} {len(cur)}->{len(fresh)}",
                          sha=r["sha"], branch="main")
    rb = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    rb_n = len(json.loads(base64.b64decode(rb["content"])))
    print(f"{lab}: {len(cur)}->{len(fresh)} (+{len(added)} -{len(removed)}) "
          f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={rb_n} sha={rb['sha'][:12]}")
    results[lab] = {"n": len(fresh), "changed": True, "added": len(added),
                    "removed": len(removed),
                    "commit": put.get("commit", {}).get("sha", "?")[:12],
                    "readback": rb_n, "sha": rb["sha"][:12]}

for lab, (sid, want) in LIVE_ONLY.items():
    fresh = pull(sid)
    wins = 0
    for m in fresh:
        w = m.get("winner")
        if (w == "a" and m.get("name_a") == want) or (w == "b" and m.get("name_b") == want):
            wins += 1
    print(f"{lab}: live-only {len(fresh)} games {wins} wins (no put)")
    results[lab] = {"n": len(fresh), "wins": wins, "live_only": True}

for lab in ["krypton-search256", "krypton-prosearch", "krypton-search256-v2"]:
    r = gh_api.get_file("robertyang87", "carbon-league", f"data/{lab}_matches.json", ref="main")
    n = len(json.loads(base64.b64decode(r["content"])))
    print(f"{lab}: verify-only n={n} sha={r['sha'][:12]}")
    results[lab] = {"n": n, "changed": False, "sha": r["sha"][:12], "verify_only": True}

json.dump(results, open("/tmp/refresh_1126_followup.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
