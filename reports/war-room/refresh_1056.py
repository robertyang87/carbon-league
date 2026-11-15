#!/usr/bin/env python3
"""newface-watch 10:56 run: 117-end match-list refresh vs repo main.
Reads each data/<label>_matches.json from main, pulls the 117-end full
list, diffs id sets, put_file only on change, get_file read-back verify.
it570 (m9xj570) is a FROZEN 89-game file: read-only sha check, never overwrite.
Prints a JSON summary to stdout."""
import base64, json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
LABELS = [
    ("m9xj630", "RyanX-AIPowercore-M9XJ-630", "b3fa633052e17329f34054bab0090128"),
    ("m3s3z", "claude-AlphaCarbon-Master3s3z", "f9fdbce38def1a78eb9c77f8711a1103"),
    ("m9xj600", "RyanX-AIPowercore-M9XJ-600", "b1cde1fda59f9bab81e56f40de92302d"),
    ("m9xj570", "RyanX-AIPowercore-M9XJ-it570", "28c19bba51a260bbae29ea5fba59951c"),
    ("xiaofeiyu5", "小肥鱼v5", "2856c37287900ee3cba8eca53b827da9"),
    ("hakimi3pro", "哈基米 v3pro", "aff6e57c5eba8857d5546783c7d39a46"),
    ("k4v3", "stephen-K4TacSearch-v3", "9d596068f39e0b8cca317c3bb27f8ae5"),
    ("fable6", "Fable5.2-Max-V6", "5b453cb4f7c49439673deda697d46a9b"),
    ("k4v2", "stephen-K4TacSearch-v2", "aaa02959b15d57e4e28f80d665ae3482"),
    ("lite-v8", "RyanX-AIPowercore-Lite-V8", "3a4d267d9fcea50794f48d38d3e1360d"),
    ("krypton-cloudsearch", "Krypton-CloudSearch-v1", "6e830c045157f572ad460c4a8a37b822"),
    ("krypton-csplant", "Krypton-CS-Plant-v1", "adaf275d51a177b7a9cf8224ab66c467"),
    ("krypton-m128", "Krypton-M128Search-v1", "96db8cfd61a73a534807cc404d046e31"),
    ("krypton", "Krypton-Cloud500", "0f2c23daf04aaf650576575d2482b940"),
    ("fable", "Fable5.2-Max-V5", None),
    ("m9xj", "RyanX-AIPowercore-M9XJ-it260", None),
    ("hakimi31", "哈基米 v3.1", None),
    ("m3s3", "claude-AlphaCarbon-Master3s3", None),
    ("krypton-search256", "Krypton-Search256-v1", None),
    ("krypton-prosearch", "Krypton-ProSearch-v1", None),
    ("krypton-search256-v2", "Krypton-Search256-v2", None),
]
TMP = "/tmp/m1056"
os.makedirs(TMP, exist_ok=True)
env = dict(os.environ, LADDER_BASE="https://117.72.210.134")

def ids_of(matches):
    return {(m.get("id"), m.get("game")) for m in matches}

def repo_matches(label):
    try:
        r = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
    except Exception as e:
        return None, None, f"get_file error: {e}"
    if isinstance(r, dict) and r.get("type") == "file":
        raw = base64.b64decode(r["content"])
        return json.loads(raw), r.get("sha"), None
    return None, None, f"unexpected get_file: {str(r)[:200]}"

def extract_sid(matches, name):
    for m in matches:
        for s in ("a", "b"):
            st = m.get("strategies", {}).get(s, {})
            if st.get("name") == name:
                return st.get("id")
    return None

summary = {}
for label, name, sid in LABELS:
    row = {"name": name}
    repo, sha, err = repo_matches(label)
    if err:
        row["error"] = err
        summary[label] = row
        continue
    row["repo_count"] = len(repo)
    row["repo_sha"] = (sha or "")[:12]
    if sid is None:
        sid = extract_sid(repo, name)
        row["sid_resolved"] = sid
    if not sid:
        row["error"] = "sid not resolved"
        summary[label] = row
        continue
    out = f"{TMP}/{label}.json"
    p = subprocess.run(["python3", "ladder.py", "matches", sid, out],
                       cwd="/home/hatch/workspace/carbon-live",
                       env=env, capture_output=True, text=True, timeout=300)
    if p.returncode != 0 or not os.path.exists(out):
        row["error"] = f"117 pull failed rc={p.returncode}: {p.stderr[-300:]}"
        summary[label] = row
        continue
    live = json.load(open(out))
    row["live_count"] = len(live)
    old_ids, new_ids = ids_of(repo), ids_of(live)
    row["added"] = len(new_ids - old_ids)
    row["removed"] = len(old_ids - new_ids)
    if label == "m9xj570":
        row["frozen"] = True
        row["note"] = "frozen file, sha check only"
    elif old_ids != new_ids:
        content = json.dumps(live, indent=1, ensure_ascii=False).encode()
        try:
            put = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                                  content,
                                  f"newface-watch 1056: {label} {len(repo)}->{len(live)}",
                                  sha=sha, branch="main")
            row["commit"] = (put.get("commit") or {}).get("sha", "")[:12]
        except Exception as e:
            row["error"] = f"put_file failed: {e}"
            summary[label] = row
            continue
        back, sha2, err2 = repo_matches(label)
        row["readback_count"] = None if err2 else len(back)
        row["readback_sha"] = (sha2 or "")[:12]
        row["readback_err"] = err2
    summary[label] = row

# corpus counts from one recursive tree read
try:
    tree = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/git/trees/main?recursive=1")
    corpus = {}
    for t in tree.get("tree", []):
        pth = t.get("path", "")
        if pth.startswith("data/") and pth.endswith("_corpus") is False:
            parts = pth.split("/")
            if len(parts) == 3 and parts[2] != "" and parts[1].endswith("_corpus"):
                corpus[parts[1][:-7]] = corpus.get(parts[1][:-7], 0) + 1
    summary["_corpus"] = corpus
except Exception as e:
    summary["_corpus_error"] = str(e)

print(json.dumps(summary, ensure_ascii=False, indent=1))
