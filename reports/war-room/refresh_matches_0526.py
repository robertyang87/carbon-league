#!/usr/bin/env python3
"""Newface-watch matches refresh 2026-11-14 05:26 run.
Pulls 117-end (https://117.72.210.134) paginated matches per label,
compares id sets vs robertyang87/carbon-league main data/<label>_matches.json,
put_file only on change, get_file read-back to verify. it570 frozen: verify only.
"""
import base64, json, ssl, sys, time, urllib.request

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
HOST = "https://117.72.210.134"

TARGETS = [
    # label, hex id, target name, repo path, frozen?
    ("m3s3z", "f9fdbce38def1a78eb9c77f8711a1103", "claude-AlphaCarbon-Master3s3z", "data/m3s3z_matches.json", False),
    ("hakimi3pro", "aff6e57c5eba8857d5546783c7d39a46", "哈基米 v3pro", "data/hakimi3pro_matches.json", False),
    ("k4v3", "9d596068f39e0b8cca317c3bb27f8ae5", "stephen-K4TacSearch-v3", "data/k4v3_matches.json", False),
    ("fable6", "5b453cb4f7c49439673deda697d46a9b", "Fable5.2-Max-V6", "data/fable6_matches.json", False),
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d", "RyanX-AIPowercore-Lite-V8", "data/lite-v8_matches.json", False),
    ("krypton", "0f2c23daf04aaf650576575d2482b940", "Krypton-Cloud500", "data/krypton_matches.json", False),
    ("fable", "2b4b44fd148fa7fdaa53bb0d110ae705", "Fable5.2-Max-V5", "data/fable_matches.json", False),
    ("m9xj570", "28c19bba51a260bbae29ea5fba59951c", "RyanX-AIPowercore-M9XJ-it570", "data/m9xj570_matches.json", True),
]

def pull117(hexid):
    got, off = [], 0
    while True:
        u = f"{HOST}/api/matches?compact=1&limit=100&offset={off}&strategy={hexid}"
        r = urllib.request.urlopen(urllib.request.Request(u), context=CTX, timeout=30)
        d = json.loads(r.read())
        got += d.get("matches", [])
        if d.get("next_offset") is None:
            break
        off = d["next_offset"]
        time.sleep(0.7)
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq

def wins_for(uniq, name):
    w = 0
    for m in uniq:
        side = "a" if m.get("name_a") == name else ("b" if m.get("name_b") == name else None)
        if side and m.get("winner") == side:
            w += 1
    return w

def repo_load(path):
    f = gh_api.get_file("robertyang87", "carbon-league", path, ref="main")
    raw = base64.b64decode(f["content"])
    return f.get("sha"), json.loads(raw)

out = {}
for label, hexid, name, path, frozen in TARGETS:
    try:
        uniq = pull117(hexid)
    except Exception as e:
        out[label] = {"error": f"117 pull failed: {e!r}"}
        print(label, "117 PULL FAILED", repr(e))
        continue
    ids = {(m.get("id"), m.get("game")) for m in uniq}
    w = wins_for(uniq, name)
    try:
        sha, repo = repo_load(path)
    except Exception as e:
        out[label] = {"error": f"repo get failed: {e!r}", "n117": len(uniq), "wins117": w}
        print(label, "REPO GET FAILED", repr(e))
        continue
    repo_ids = {(m.get("id"), m.get("game")) for m in repo}
    rec = {"n117": len(uniq), "wins117": w, "nrepo": len(repo),
           "repo_sha": sha[:12], "id_diff": len(ids ^ repo_ids)}
    if frozen:
        rec["frozen"] = True
        rec["action"] = "verify-only"
        print(label, f"117={len(uniq)}/{w} repo={len(repo)} sha={sha[:12]} FROZEN verify-only")
    elif ids == repo_ids:
        rec["action"] = "no-change"
        print(label, f"117={len(uniq)}/{w} repo={len(repo)} NO CHANGE")
    else:
        body = json.dumps(uniq, indent=1, ensure_ascii=False).encode()
        put = gh_api.put_file("robertyang87", "carbon-league", path, body,
                              f"newface-watch: refresh {label} matches {len(repo)}->{len(uniq)}",
                              sha=sha, branch="main")
        sha2, repo2 = repo_load(path)
        rec["action"] = "updated"
        rec["put_commit"] = put.get("commit", {}).get("sha", "")[:12]
        rec["readback_n"] = len(repo2)
        rec["readback_sha"] = sha2[:12]
        rec["readback_ok"] = len(repo2) == len(uniq) and {(m.get("id"), m.get("game")) for m in repo2} == ids
        print(label, f"117={len(uniq)}/{w} repo {len(repo)}->{len(uniq)} commit={rec['put_commit']} readback={len(repo2)} ok={rec['readback_ok']}")
    out[label] = rec

# vanished strategies: repo read only
for label, path in [("hakimi31", "data/hakimi31_matches.json"),
                    ("m3s3", "data/m3s3_matches.json")]:
    try:
        sha, repo = repo_load(path)
        out[label] = {"nrepo": len(repo), "repo_sha": sha[:12], "action": "vanished-verify-only"}
        print(label, f"repo={len(repo)} sha={sha[:12]} VANISHED verify-only")
    except Exception as e:
        out[label] = {"error": repr(e)}
        print(label, "REPO GET FAILED", repr(e))

json.dump(out, open("/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/refresh_matches_0526.json", "w"), ensure_ascii=False, indent=1)
print("DONE")
