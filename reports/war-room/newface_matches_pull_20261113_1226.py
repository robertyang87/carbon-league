#!/usr/bin/env python3
"""newface-watch 12:26 run: 117-end matches pull + repo compare/refresh + stats.
Labels follow repo precedent. Writes a JSON report to stdout and to the
run report file given as argv[1]."""
import base64, json, ssl, statistics, sys, time, urllib.request
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

HOST117 = "https://117.72.210.134"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

MEMBERS = [
    ("m9xj570", "RyanX-AIPowercore-M9XJ-it570", "28c19bba51a260bbae29ea5fba59951c"),
    ("krypton", "Krypton-Cloud500", "0f2c23daf04aaf650576575d2482b940"),
    ("lite-v8", "RyanX-AIPowercore-Lite-V8", "3a4d267d9fcea50794f48d38d3e1360d"),
    ("hakimi3", "哈基米 v3", "c6c24a9d627fabf6322ce8d7c3ace6f3"),
    ("fable", "Fable5.2-Max-V5", "2b4b44fd148fa7fdaa53bb0d110ae705"),
    ("lite-v7", "RyanX-AIPowercore-Lite-V7", "694c1fd3a79473b08354c6f7364a3237"),
    ("godslayer_cvi", "AgyCarbon-P2C-Godslayer-CVI", "c1c405519a1a04962c2d38a343882fab"),
    ("m9xj320", "RyanX-AIPowercore-M9XJ-it320", "28cd4ac93ab020aa6c1abd821675339d"),
]
OWNER, REPO = "robertyang87", "carbon-league"

def pull117(sid):
    got, off = [], 0
    while True:
        url = f"{HOST117}/api/matches?compact=1&limit=100&offset={off}&strategy={sid}"
        req = urllib.request.Request(url, headers={"User-Agent": "muse-agent"})
        last_err = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60, context=CTX) as resp:
                    d = json.loads(resp.read())
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        if last_err is not None:
            raise last_err
        got += d.get("matches", [])
        if d.get("next_offset") is None:
            break
        off = d["next_offset"]
        time.sleep(0.5)
    seen, uniq = set(), []
    for m in got:
        k = (m.get("id"), m.get("game"))
        if k not in seen:
            seen.add(k)
            uniq.append(m)
    uniq.sort(key=lambda m: m.get("created", 0))
    return uniq

def stats(ms, name):
    if not ms:
        return {"total": 0}
    wins = 0
    margins, first_w, first_n, second_w, second_n = [], 0, 0, 0, 0
    opp = {}
    for m in ms:
        side = "a" if m.get("name_a") == name else "b"
        other = "b" if side == "a" else "a"
        w = (m.get("winner") == side)
        wins += 1 if w else 0
        sc = m.get("scores", {})
        margins.append(round(sc.get(side, 0) - sc.get(other, 0), 1))
        is_first = bool(m.get("order")) and m["order"][0] == side
        if is_first:
            first_n += 1
            first_w += 1 if w else 0
        else:
            second_n += 1
            second_w += 1 if w else 0
        oname = m.get("name_a" if other == "a" else "name_b")
        o = opp.setdefault(oname, [0, 0])
        o[0] += 1 if w else 0
        o[1] += 1
    last10 = ms[-10:]
    lw = sum(1 for m in last10 if m.get("winner") == ("a" if m.get("name_a") == name else "b"))
    lm = []
    for m in last10:
        side = "a" if m.get("name_a") == name else "b"
        other = "b" if side == "a" else "a"
        sc = m.get("scores", {})
        lm.append(round(sc.get(side, 0) - sc.get(other, 0), 1))
    return {
        "total": len(ms), "wins": wins,
        "winrate": round(100 * wins / len(ms), 1),
        "mean_margin": round(statistics.mean(margins), 1),
        "median_margin": round(statistics.median(margins), 1),
        "first": f"{first_w}/{first_n}", "second": f"{second_w}/{second_n}",
        "last10_wins": lw, "last10_mean_margin": round(statistics.mean(lm), 1),
        "first_created": ms[0].get("created"), "last_created": ms[-1].get("created"),
        "opponents": {k: f"{v[0]}/{v[1]}" for k, v in sorted(opp.items(), key=lambda kv: -kv[1][1])[:12]},
    }

def repo_file(label):
    try:
        r = gh_api.get_file(OWNER, REPO, f"data/{label}_matches.json")
        raw = base64.b64decode(r["content"]).decode()
        return json.loads(raw), r.get("sha")
    except Exception as e:
        return None, None

def corpus_count(label):
    try:
        r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/data/{label}_corpus")
        if isinstance(r, list):
            return len([x for x in r if x.get("type") == "file"])
        return None
    except Exception:
        return None

report = {"members": {}, "corpus": {}, "workflow": None}
for label, name, sid in MEMBERS:
    fresh = pull117(sid)
    json.dump(fresh, open(f"/home/hatch/workspace/carbon-live/matches_{sid}.json", "w"),
              indent=1, ensure_ascii=False)
    old, sha = repo_file(label)
    old_ids = {m.get("id") for m in old} if old else set()
    fresh_ids = {m.get("id") for m in fresh}
    new_ids = fresh_ids - old_ids
    entry = {"total": len(fresh), "repo_total": len(old) if old is not None else None,
             "new_ids": len(new_ids), "sha_before": (sha or "")[:12],
             "stats": stats(fresh, name), "action": "unchanged"}
    if old is None:
        entry["action"] = "created"
        r = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                            json.dumps(fresh, ensure_ascii=False, indent=1).encode(),
                            f"newface-watch: add {label} matches ({len(fresh)})")
        entry["put_sha"] = r.get("content", {}).get("sha", "")[:12]
    elif new_ids:
        entry["action"] = "updated"
        r = gh_api.put_file(OWNER, REPO, f"data/{label}_matches.json",
                            json.dumps(fresh, ensure_ascii=False, indent=1).encode(),
                            f"newface-watch: {label} matches {len(old)}->{len(fresh)}",
                            sha=sha)
        entry["put_sha"] = r.get("content", {}).get("sha", "")[:12]
    # read-back verify
    back, sha2 = repo_file(label)
    entry["readback_total"] = len(back) if back is not None else None
    entry["readback_sha"] = (sha2 or "")[:12]
    entry["verified"] = (entry["readback_total"] == len(fresh))
    report["members"][label] = entry
    print(label, entry["action"], entry["total"], "repo:", entry["repo_total"],
          "readback:", entry["readback_total"], "verified:", entry["verified"], flush=True)

for label in ["m9xj570", "krypton", "lite-v8", "hakimi3", "lite-v7", "m9xj320", "fable",
              "godslayer_cvi"]:
    report["corpus"][label] = corpus_count(label)
    print("corpus", label, report["corpus"][label], flush=True)

try:
    runs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/ladder-replay-pull.yml/runs?per_page=5")
    rs = runs.get("workflow_runs", []) if isinstance(runs, dict) else []
    report["workflow"] = [{"id": r["id"], "status": r["status"], "conclusion": r.get("conclusion"),
                           "created_at": r["created_at"]} for r in rs]
    print("workflow runs:", json.dumps(report["workflow"], ensure_ascii=False), flush=True)
except Exception as e:
    report["workflow"] = f"ERROR {e}"
    print("workflow ERROR", e, flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/newface_pull_report.json"
json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
print("REPORT", out)
