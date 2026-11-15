#!/usr/bin/env python3
"""newface-watch 12:56 run: 117-end match-list refresh vs repo main.
Combines refresh_1056.py + refresh_1056_followup.py, plus master4s (new
label, created 11:56) and m9xj320/it320 (verify-only, handed over).
Reads each data/<label>_matches.json from main, pulls the 117-end full
list via ladder.py (LADDER_BASE=117), diffs id sets, put_file only on
change (except frozen m9xj570 and verify-only labels), get_file read-back
verify. Name guard on every pull: every match must name the target.
Prints a JSON summary to stdout."""
import base64, json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
# (label, name, sid, mode)  mode: put | frozen | verify
LABELS = [
    ("master4s", "claude-AlphaCarbon-Master4s", "6389c346461bce295badcf52a1dd74c6", "put"),
    ("m9xj630", "RyanX-AIPowercore-M9XJ-630", "b3fa633052e17329f34054bab0090128", "put"),
    ("xiaofeiyu5", "小肥鱼v5", "2856c37287900ee3cba8eca53b827da9", "put"),
    ("m9xj600", "RyanX-AIPowercore-M9XJ-600", "b1cde1fda59f9bab81e56f40de92302d", "put"),
    ("m3s3z", "claude-AlphaCarbon-Master3s3z", "f9fdbce38def1a78eb9c77f8711a1103", "put"),
    ("hakimi3pro", "哈基米 v3pro", "aff6e57c5eba8857d5546783c7d39a46", "put"),
    ("k4v3", "stephen-K4TacSearch-v3", "9d596068f39e0b8cca317c3bb27f8ae5", "put"),
    ("fable6", "Fable5.2-Max-V6", "5b453cb4f7c49439673deda697d46a9b", "put"),
    ("k4v2", "stephen-K4TacSearch-v2", "aaa02959b15d57e4e28f80d665ae3482", "put"),
    ("lite-v8", "RyanX-AIPowercore-Lite-V8", "3a4d267d9fcea50794f48d38d3e1360d", "put"),
    ("krypton-cloudsearch", "Krypton-CloudSearch-v1", "6e830c045157f572ad460c4a8a37b822", "put"),
    ("krypton-csplant", "Krypton-CS-Plant-v1", "adaf275d51a177b7a9cf8224ab66c467", "put"),
    ("krypton-m128", "Krypton-M128Search-v1", "96db8cfd61a73a534807cc404d046e31", "put"),
    ("krypton", "Krypton-Cloud500", "0f2c23daf04aaf650576575d2482b940", "put"),
    ("fable", "Fable5.2-Max-V5", "2b4b44fd148fa7fdaa53bb0d110ae705", "put"),
    ("m9xj570", "RyanX-AIPowercore-M9XJ-it570", "28c19bba51a260bbae29ea5fba59951c", "frozen"),
    ("m9xj320", "RyanX-AIPowercore-M9XJ-it320", "28cd4ac93ab020aa6c1abd821675339d", "verify"),
    ("m9xj", "RyanX-AIPowercore-M9XJ-it260", "e43b20c5d8d644b4145cb1e7ff198201", "verify"),
    ("hakimi31", "哈基米 v3.1", "29844f8f0c9b14b3d37902a551d26a1e", "verify"),
    ("m3s3", "claude-AlphaCarbon-Master3s3", "f47e13ba624a85c5c2dc1b40caecc193", "verify"),
    ("krypton-search256", "Krypton-Search256-v1", "2c57e7cd01dee46bef98643398ab4ca3", "put"),
    ("krypton-prosearch", "Krypton-ProSearch-v1", "fede68b79a2965889eeae0e226601139", "put"),
    ("krypton-search256-v2", "Krypton-Search256-v2", "484814899c333f8149e984eacdaf5275", "put"),
    ("fable7", "Fable5.2-Max-V7", "6b14b59a4a5c2e6850cee5a471400340", "conditional"),
]
FROZEN_M9XJ570_SHA_PREFIX = "9d69b8472d70"
TMP = "/tmp/m1256"
os.makedirs(TMP, exist_ok=True)
env = dict(os.environ, LADDER_BASE="https://117.72.210.134")

def repo_matches(label):
    try:
        r = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
    except Exception as e:
        return None, None, f"get_file error: {e}"
    if isinstance(r, dict) and r.get("type") == "file":
        return json.loads(base64.b64decode(r["content"])), r.get("sha"), None
    return None, None, f"unexpected get_file: {str(r)[:200]}"

summary = {}
for label, name, sid, mode in LABELS:
    row = {"name": name, "mode": mode}
    repo, sha, err = repo_matches(label)
    if err:
        if mode == "conditional":
            row["note"] = "no repo file, skipped"
            summary[label] = row
            continue
        row["error"] = err
        summary[label] = row
        continue
    row["repo_count"] = len(repo)
    row["repo_sha"] = (sha or "")[:12]
    out = f"{TMP}/{label}.json"
    p = subprocess.run(["python3", "ladder.py", "matches", sid, out],
                       cwd="/home/hatch/workspace/carbon-live",
                       env=env, capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not os.path.exists(out):
        row["error"] = f"117 pull failed rc={p.returncode}: {p.stderr[-300:]}"
        summary[label] = row
        continue
    live = json.load(open(out))
    bad = [m for m in live if name not in (m.get("name_a"), m.get("name_b"))]
    row["live_count"] = len(live)
    row["name_guard_bad"] = len(bad)
    if bad:
        row["error"] = "NAME GUARD FAIL, skipped"
        summary[label] = row
        continue
    wins = sum(1 for m in live if (m.get("winner") == "a" and m.get("name_a") == name) or
               (m.get("winner") == "b" and m.get("name_b") == name))
    row["live_wins"] = wins
    old_ids = {(m.get("id"), m.get("game")) for m in repo}
    new_ids = {(m.get("id"), m.get("game")) for m in live}
    row["added"] = len(new_ids - old_ids)
    row["removed"] = len(old_ids - new_ids)
    if mode == "frozen":
        row["frozen_sha_ok"] = (sha or "").startswith(FROZEN_M9XJ570_SHA_PREFIX)
        row["note"] = "frozen file, sha check only"
    elif mode == "verify":
        row["note"] = "verify-only, no put"
    elif mode == "conditional":
        row["note"] = "conditional label, put on change"
        if old_ids != new_ids:
            try:
                put = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                                      json.dumps(live, indent=1, ensure_ascii=False).encode(),
                                      f"newface-watch 1256: {label} {len(repo)}->{len(live)}",
                                      sha=sha, branch="main")
                row["commit"] = (put.get("commit") or {}).get("sha", "")[:12]
            except Exception as e:
                row["error"] = f"put_file failed: {e}"
                summary[label] = row
                continue
            r2 = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
            row["readback_count"] = len(json.loads(base64.b64decode(r2["content"])))
            row["readback_sha"] = (r2.get("sha") or "")[:12]
    elif old_ids != new_ids:
        try:
            put = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                                  json.dumps(live, indent=1, ensure_ascii=False).encode(),
                                  f"newface-watch 1256: {label} {len(repo)}->{len(live)}",
                                  sha=sha, branch="main")
            row["commit"] = (put.get("commit") or {}).get("sha", "")[:12]
        except Exception as e:
            row["error"] = f"put_file failed: {e}"
            summary[label] = row
            continue
        r2 = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
        row["readback_count"] = len(json.loads(base64.b64decode(r2["content"])))
        row["readback_sha"] = (r2.get("sha") or "")[:12]
    summary[label] = row

# corpus counts from one recursive tree read
try:
    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/main?recursive=1")
    corpus = {}
    for t in tree.get("tree", []):
        pth = t.get("path", "")
        parts = pth.split("/")
        if len(parts) == 3 and parts[0] == "data" and parts[1].endswith("_corpus") and parts[2] != "":
            corpus[parts[1][:-7]] = corpus.get(parts[1][:-7], 0) + 1
    summary["_corpus"] = corpus
except Exception as e:
    summary["_corpus_error"] = str(e)

print(json.dumps(summary, ensure_ascii=False, indent=1))
