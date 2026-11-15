#!/usr/bin/env python3
"""Retry for 10:26 run: three labels whose ladder.py pull died on
IncompleteRead (lite-v8, k4v2, fable). Pull via curl -skL paginated
117-end /api/matches (compact=1, limit=100), assemble the same list
format ladder.py produces, then compare id sets with warehouse main and
put_file only on change, with read-back verification and name guard."""
import base64, json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

WORK = "/tmp/newface_1026"
OWNER, REPO = "robertyang87", "carbon-league"
TARGETS = [
    ("lite-v8", "3a4d267d9fcea50794f48d38d3e1360d", "data/lite-v8_matches.json", "RyanX-AIPowercore-Lite-V8"),
    ("k4v2",    "aaa02959b15d57e4e28f80d665ae3482", "data/k4v2_matches.json", "stephen-K4TacSearch-v2"),
    ("fable",   "2b4b44fd148fa7fdaa53bb0d110ae705", "data/fable_matches.json", "Fable5.2-Max-V5"),
]

def curl_pull(sid):
    out, offset = [], 0
    while True:
        url = (f"https://117.72.210.134/api/matches?compact=1&limit=100"
               f"&offset={offset}&strategy={sid}")
        r = subprocess.run(["curl", "-skL", "--max-time", "120", url],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(f"curl rc={r.returncode} {r.stderr.strip()[-150:]}")
        d = json.loads(r.stdout)
        out.extend(d.get("matches", []))
        nxt = d.get("next_offset")
        if not nxt:
            break
        offset = nxt
    return out

def ids(ms):
    return {(m.get("id"), m.get("game")) for m in ms}

report = []
for label, sid, path, name in TARGETS:
    try:
        new = curl_pull(sid)
    except Exception as e:
        report.append(f"{label}: CURL PULL FAILED {e}")
        continue
    json.dump(new, open(f"{WORK}/{label}_matches_117.json", "w"), ensure_ascii=False)
    bad = [m for m in new if name not in (m.get("name_a"), m.get("name_b"))]
    if bad:
        report.append(f"{label}: NAME GUARD FAIL {len(bad)}/{len(new)} -- SKIPPED")
        continue
    wins = sum(1 for m in new if (m.get("winner") == "a" and m.get("name_a") == name) or (m.get("winner") == "b" and m.get("name_b") == name))
    new_ids = ids(new)
    cur = gh_api.get_file(OWNER, REPO, path, ref="main")
    old = json.loads(base64.b64decode(cur["content"]).decode())
    old_ids = ids(old)
    if new_ids == old_ids:
        report.append(f"{label}: no change ({len(new)} games, {wins} wins, id set identical)")
        continue
    added = new_ids - old_ids
    removed = old_ids - new_ids
    content = open(f"{WORK}/{label}_matches_117.json", "rb").read()
    put = gh_api.put_file(OWNER, REPO, path, content,
                          f"newface-watch 10:26: {label} matches {len(old)}->{len(new)}",
                          sha=cur["sha"], branch="main")
    back = gh_api.get_file(OWNER, REPO, path, ref="main")
    back_n = len(json.loads(base64.b64decode(back["content"]).decode())) if back else -1
    report.append(f"{label}: {len(old)}->{len(new)} wins={wins} (+{len(added)} -{len(removed)}) "
                  f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={back_n} "
                  f"sha={back['sha'][:12] if back else '?'} "
                  f"{'OK' if back_n == len(new) else 'READBACK MISMATCH!'}")

print("\n".join(report))
json.dump(report, open(f"{WORK}/retry_report.json", "w"), ensure_ascii=False, indent=1)
