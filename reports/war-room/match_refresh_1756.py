#!/usr/bin/env python3
"""2b match-list refresh, run 17:56 2026-11-14.
117-end full lists via curl (matches117.py logic, robust vs ladder.py
IncompleteRead), compare id sets with warehouse main data/<label>_matches.json,
put_file only on change, get_file read-back verify. it570 FROZEN at 89
(sha 9d69b8472d70): pull for reporting only, never update. Name guard: every
pulled match must contain the target name as name_a or name_b, else skip
without submitting (lesson of the 09:56 m9xj wrong-ID incident).
m9xj (it260, delisted), hakimi31, m3s3: verify-only, never put.
Also counts corpus files via GitHub contents API for gap computation."""
import base64, json, os, subprocess, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

WORK = "/tmp/newface_1756"
os.makedirs(WORK, exist_ok=True)
OWNER, REPO = "robertyang87", "carbon-league"

TARGETS = [
    ("m3s3z",     "f9fdbce38def1a78eb9c77f8711a1103", "data/m3s3z_matches.json", "claude-AlphaCarbon-Master3s3z", False),
    ("hakimi3pro","aff6e57c5eba8857d5546783c7d39a46", "data/hakimi3pro_matches.json", "哈基米 v3pro", False),
    ("k4v3",      "9d596068f39e0b8cca317c3bb27f8ae5", "data/k4v3_matches.json", "stephen-K4TacSearch-v3", False),
    ("fable6",    "5b453cb4f7c49439673deda697d46a9b", "data/fable6_matches.json", "Fable5.2-Max-V6", False),
    ("lite-v8",   "3a4d267d9fcea50794f48d38d3e1360d", "data/lite-v8_matches.json", "RyanX-AIPowercore-Lite-V8", False),
    ("krypton",   "0f2c23daf04aaf650576575d2482b940", "data/krypton_matches.json", "Krypton-Cloud500", False),
    ("k4v2",      "aaa02959b15d57e4e28f80d665ae3482", "data/k4v2_matches.json", "stephen-K4TacSearch-v2", False),
    ("fable",     "2b4b44fd148fa7fdaa53bb0d110ae705", "data/fable_matches.json", "Fable5.2-Max-V5", False),
    ("krypton-search256", "2c57e7cd01dee46bef98643398ab4ca3", "data/krypton-search256_matches.json", "Krypton-Search256-v1", False),
    ("krypton-prosearch", "fede68b79a2965889eeae0e226601139", "data/krypton-prosearch_matches.json", "Krypton-ProSearch-v1", False),
    ("m9xj570",   "28c19bba51a260bbae29ea5fba59951c", "data/m9xj570_matches.json", "RyanX-AIPowercore-M9XJ-it570", True),
]
VERIFY_ONLY = [
    ("m9xj", "e43b20c5d8d644b4145cb1e7ff198201", "data/m9xj_matches.json", "RyanX-AIPowercore-M9XJ-it260"),
    ("hakimi31", "29844f8f0c9b14b3d37902a551d26a1e", "data/hakimi31_matches.json", "哈基米 v3.1"),
    ("m3s3", "f47e13ba624a85c5c2dc1b40caecc193", "data/m3s3_matches.json", "claude-AlphaCarbon-Master3s3"),
]

def pull117(sid):
    got, off = [], 0
    while True:
        url = (f"https://117.72.210.134/api/matches?compact=1&limit=100"
               f"&offset={off}&strategy={sid}")
        r = subprocess.run(["curl", "-skL", "--max-time", "60", url],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"curl rc={r.returncode} offset={off}: {r.stderr[:200]}")
        d = json.loads(r.stdout)
        items = d.get("matches") or d.get("items") or []
        if isinstance(d, list):
            items = d
        got.extend(items)
        nxt = d.get("next_offset") if isinstance(d, dict) else None
        if nxt is None or nxt == "" or nxt == off:
            break
        off = nxt
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    return uniq

def ids(ms):
    return {(m.get("id"), m.get("game")) for m in ms}

def corpus_count(label):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/data/{label}_corpus")
        if isinstance(r, list):
            return len([e for e in r if e.get("type") == "file"])
        return f"NOT-A-LIST:{type(r).__name__}"
    except Exception as e:
        return f"ERR:{e}"

def wins_of(ms, name):
    return sum(1 for m in ms
               if (m.get("winner") == "a" and m.get("name_a") == name)
               or (m.get("winner") == "b" and m.get("name_b") == name))

report = []
pulled = {}
for label, sid, path, name, frozen in TARGETS:
    try:
        new = pull117(sid)
    except Exception as e:
        report.append(f"{label}: PULL FAILED {e}")
        continue
    pulled[label] = new
    bad = [m for m in new if name not in (m.get("name_a"), m.get("name_b"))]
    if bad:
        report.append(f"{label}: NAME GUARD FAIL {len(bad)}/{len(new)} matches lack {name} -- SKIPPED, no submit")
        continue
    wins = wins_of(new, name)
    new_ids = ids(new)
    cur = gh_api.get_file(OWNER, REPO, path, ref="main")
    if cur is None:
        report.append(f"{label}: warehouse file MISSING ({len(new)} pulled, {wins} wins)")
        continue
    old = json.loads(base64.b64decode(cur["content"]).decode())
    old_ids = ids(old)
    sha = cur["sha"]
    if frozen:
        report.append(f"{label}: FROZEN pull={len(new)} wins={wins} ware={len(old)} sha={sha[:12]} "
                      f"{'UNCHANGED' if sha.startswith('9d69b8472d70') else 'SHA CHANGED!'}")
        continue
    if new_ids == old_ids:
        report.append(f"{label}: no change ({len(new)} games, {wins} wins, id set identical)")
        continue
    added = new_ids - old_ids
    removed = old_ids - new_ids
    out = f"{WORK}/{label}_matches_117.json"
    json.dump(new, open(out, "w"), ensure_ascii=False)
    content = open(out, "rb").read()
    put = gh_api.put_file(OWNER, REPO, path, content,
                          f"newface-watch 17:56: {label} matches {len(old)}->{len(new)}",
                          sha=sha, branch="main")
    back = gh_api.get_file(OWNER, REPO, path, ref="main")
    back_n = len(json.loads(base64.b64decode(back["content"]).decode())) if back else -1
    report.append(f"{label}: {len(old)}->{len(new)} wins={wins} (+{len(added)} -{len(removed)}) "
                  f"commit={put.get('commit',{}).get('sha','?')[:12]} readback={back_n} "
                  f"sha={back['sha'][:12] if back else '?'} "
                  f"{'OK' if back_n == len(new) else 'READBACK MISMATCH!'}")

for label, sid, path, name in VERIFY_ONLY:
    try:
        new = pull117(sid)
    except Exception as e:
        report.append(f"{label}: PULL FAILED {e}")
        continue
    bad = [m for m in new if name not in (m.get("name_a"), m.get("name_b"))]
    if bad:
        report.append(f"{label}: NAME GUARD FAIL {len(bad)}/{len(new)} -- read-only, no action")
        continue
    cur = gh_api.get_file(OWNER, REPO, path, ref="main")
    old = json.loads(base64.b64decode(cur["content"]).decode())
    report.append(f"{label}: VERIFY-ONLY pull={len(new)} ware={len(old)} "
                  f"idset={'IDENTICAL' if ids(new) == ids(old) else 'DIFFERS!'} sha={cur['sha'][:12]}")

corpus = {}
for label in ["m3s3z", "hakimi3pro", "k4v3", "fable6", "lite-v8", "krypton",
              "k4v2", "fable", "krypton-search256", "krypton-prosearch",
              "m9xj570", "hakimi31", "m3s3"]:
    corpus[label] = corpus_count(label)
    report.append(f"corpus {label}: {corpus[label]}")

print("\n".join(report))
json.dump({"report": report,
           "pulled_counts": {k: len(v) for k, v in pulled.items()},
           "corpus": corpus},
          open(f"{WORK}/refresh_report.json", "w"), ensure_ascii=False, indent=1)
