#!/usr/bin/env python3
"""newface-watch 21:56 run (this run, written fresh): 117-end matches pull
+ repo compare/refresh for watchlist members. Task-body rules:
- 117 endpoint is the full view; board numbers are lagging hints only.
- m9xj570: VERIFY-ONLY (frozen at 89 by owner ruling 2026-11-13 20:31; never put).
- m3s3 / lite-v7 / m9xj320: vanished members; verify counts, refresh only if
  the 117 list actually grew (final-pull insurance).
- Others: put_file only when the 117 list differs from main; read back after put.
Writes a JSON report to argv[1]."""
import json, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST117 = "https://117.72.210.134"
OWNER, REPO = "robertyang87", "carbon-league"

MEMBERS = [
    ("m9xj570", "RyanX-AIPowercore-M9XJ-it570", "28c19bba51a260bbae29ea5fba59951c", True),
    ("krypton", "Krypton-Cloud500", "0f2c23daf04aaf650576575d2482b940", False),
    ("lite-v8", "RyanX-AIPowercore-Lite-V8", "3a4d267d9fcea50794f48d38d3e1360d", False),
    ("hakimi3", "哈基米 v3", "c6c24a9d627fabf6322ce8d7c3ace6f3", False),
    ("hakimi3pro", "哈基米 v3pro", "aff6e57c5eba8857d5546783c7d39a46", False),
    ("hakimi31", "哈基米 v3.1", "29844f8f0c9b14b3d37902a551d26a1e", False),
    ("k4v3", "stephen-K4TacSearch-v3", "9d596068f39e0b8cca317c3bb27f8ae5", False),
    ("m3s3z", "claude-AlphaCarbon-Master3s3z", "f9fdbce38def1a78eb9c77f8711a1103", False),
    ("m3s3", "claude-AlphaCarbon-Master3s3", "f47e13ba624a85c5c2dc1b40caecc193", False),
    ("fable6", "Fable5.2-Max-V6", "5b453cb4f7c49439673deda697d46a9b", False),
    ("fable", "Fable5.2-Max-V5", "2b4b44fd148fa7fdaa53bb0d110ae705", False),
    ("lite-v7", "RyanX-AIPowercore-Lite-V7", "694c1fd3a79473b08354c6f7364a3237", False),
    ("m9xj320", "RyanX-AIPowercore-M9XJ-it320", "28cd4ac93ab020aa6c1abd821675339d", False),
]

def pull117(sid):
    got, off, pages = [], 0, 0
    while True:
        url = f"{HOST117}/api/matches?compact=1&limit=100&offset={off}&strategy={sid}"
        r = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"curl rc={r.returncode}: {r.stderr[:200]}")
        d = json.loads(r.stdout)
        ms = d.get("matches", [])
        got.extend(ms)
        pages += 1
        nxt = d.get("next_offset")
        if not ms or nxt in (None, "", 0):
            break
        off = int(nxt)
        if pages > 30:
            raise RuntimeError("too many pages")
    return got

def stats(name, ms):
    wins = 0
    first_w = first_t = 0
    for m in ms:
        side = "a" if m["name_a"] == name else ("b" if m["name_b"] == name else None)
        if side is None:
            continue
        if m.get("winner") == side:
            wins += 1
        order = m.get("order") or []
        if order and order[0] == side:
            first_t += 1
            if m.get("winner") == side:
                first_w += 1
    return {"total": len(ms), "wins": wins,
            "first": f"{first_w}/{first_t}",
            "second": f"{wins - first_w}/{len(ms) - first_t}"}

def main():
    out_path = sys.argv[1]
    report = {}
    for label, name, sid, verify_only in MEMBERS:
        ms = pull117(sid)
        # sanity: every match must involve this strategy
        bad = [m for m in ms if name not in (m.get("name_a"), m.get("name_b"))]
        if bad:
            report[label] = {"error": f"{len(bad)} matches do not involve {name}; id suspect, skipped"}
            print(f"{label}: ERROR id suspect ({len(bad)} foreign matches)")
            continue
        json.dump(ms, open(f"/home/hatch/workspace/carbon-live/pull_{label}_2156.json", "w"), ensure_ascii=False)
        st = stats(name, ms)
        path = f"data/{label}_matches.json"
        repo = gh_api.get_file(OWNER, REPO, path, ref="main")
        if repo is None:
            repo_n, repo_ids, sha = 0, set(), None
        else:
            import base64
            content = repo.get("content") or ""
            raw = base64.b64decode("".join(content.split())) if isinstance(content, str) else b""
            try:
                repo_list = json.loads(raw.decode())
            except Exception:
                repo_list = []
            repo_n = len(repo_list)
            repo_ids = {m.get("id") for m in repo_list}
            sha = repo.get("sha")
        new_ids = {m.get("id") for m in ms}
        changed = new_ids != repo_ids
        action = "unchanged"
        if changed and not verify_only:
            body = json.dumps(ms, ensure_ascii=False, indent=1).encode()
            put = gh_api.put_file(OWNER, REPO, path, body,
                                  f"newface-watch 21:56: refresh {label} matches {repo_n}->{len(ms)}",
                                  sha=sha, branch="main")
            back = gh_api.get_file(OWNER, REPO, path, ref="main")
            import base64 as _b
            raw2 = _b.b64decode("".join((back.get("content") or "").split()))
            back_n = len(json.loads(raw2.decode()))
            action = f"put+readback {repo_n}->{len(ms)} (readback {back_n}) sha={put.get('commit',{}).get('sha') if isinstance(put, dict) else put}"
            if back_n != len(ms):
                action += " READBACK MISMATCH"
        elif changed and verify_only:
            action = f"changed {repo_n}->{len(ms)} but VERIFY-ONLY, not put"
        report[label] = {"name": name, "stats": st, "repo_n": repo_n,
                         "changed": changed, "action": action}
        print(f"{label}: 117={st['total']} wins={st['wins']} first={st['first']} second={st['second']} repo={repo_n} -> {action}")
    json.dump(report, open(out_path, "w"), ensure_ascii=False, indent=1)
    print("report ->", out_path)

if __name__ == "__main__":
    main()
