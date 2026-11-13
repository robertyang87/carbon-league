"""Carbon League public-source census, phase 2b+3.

2b: retry matches paging offsets 5200..7900 to top up enumeration.
3: serially probe GET /source/<id> for every enumerated id; save public
   Python sources as reference_<slug>.py. Incremental progress in
   census_results.json (rewritten after every probe).
"""
import json
import os
import re
import subprocess
import time

BASE = "https://132.232.176.153"
KNOWN = {
    "a938df4254388b8df8ee04f6e6e0aae5": ("juzi", "reference_juzi.py"),
    "97898ce7047c0bb1dd7a13f02eff0dac": ("claude-opus-v53", "reference_claude_opus_v53.py"),
    "6fb751529a6cd9fdd3e337a467ba1871": ("GloriousEvolution-v53-rho06", "reference_gloriousevolution_v53.py"),
}

census = json.load(open("census_ids.json"))
id2name = census["id2name"]
ratings = census["ratings"]

# ---------- phase 2b: top-up paging ----------
added = 0
for off in range(8000, 20000, 100):
    url = f"{BASE}/api/matches?limit=100&offset={off}"
    try:
        out = subprocess.run(["curl", "-sk", "--max-time", "60", url],
                             capture_output=True, text=True, timeout=90)
        d = json.loads(out.stdout)
        ms = d.get("matches", [])
    except Exception as e:
        print(f"topup offset {off} failed: {e}", flush=True)
        continue
    if not ms:
        print(f"topup offset {off}: empty", flush=True)
        continue
    for m in ms:
        for key_id, key_name in (("a", "name_a"), ("b", "name_b")):
            sid, nm = m.get(key_id), m.get(key_name)
            if sid and nm and sid not in id2name:
                id2name[sid] = nm
                added += 1
    time.sleep(0.4)
print(f"topup added {added}; total ids {len(id2name)}", flush=True)
census["id2name"] = id2name
json.dump(census, open("census_ids.json", "w"), ensure_ascii=False, indent=1)


def slugify(name, sid, used):
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    s = re.sub(r"_+", "_", s)[:60].strip("_")
    if not s:
        s = "cn_" + sid[:8]
    base = s
    if s in used:
        s = f"{base}_{sid[:6]}"
    used.add(s)
    return s


def looks_python(body):
    if not body or body.lstrip().startswith("{"):
        return False
    head = body[:400]
    return (
        body.lstrip().startswith(('"""', "'''", "#!", "#"))
        or "\nimport " in head or head.startswith(("import ", "from ", "def "))
        or "def agent" in body[:4000]
    )


results = {}  # id -> record
if os.path.exists("census_results.json"):
    results = json.load(open("census_results.json"))
used_slugs = {r["slug"] for r in results.values() if r.get("slug")}

ids = sorted(id2name.keys())
todo = [sid for sid in ids if sid not in results]
print(f"probing {len(todo)} ids ({len(ids)} total, {len(results)} already done)", flush=True)

for i, sid in enumerate(todo):
    name = id2name[sid]
    rec = {"name": name, "rating": ratings.get(sid)}
    if sid in KNOWN:
        rec.update({"status": "known", "file": KNOWN[sid][1], "slug": None})
        results[sid] = rec
        continue
    try:
        out = subprocess.run(
            ["curl", "-sk", "--max-time", "60", "-w", "\n%{http_code}", f"{BASE}/source/{sid}"],
            capture_output=True, text=True, timeout=90,
        )
        body, _, code = out.stdout.rpartition("\n")
        code = int(code) if code.isdigit() else 0
    except Exception as e:
        rec.update({"status": "error", "error": str(e)})
        results[sid] = rec
        print(f"[{i+1}/{len(todo)}] {name} ERROR {e}", flush=True)
        continue
    rec["http"] = code
    if code == 200 and looks_python(body):
        slug = slugify(name, sid, used_slugs)
        fn = f"reference_{slug}.py"
        with open(fn, "w") as f:
            f.write(body)
        rec.update({"status": "public", "file": fn, "slug": slug, "bytes": len(body.encode())})
        print(f"[{i+1}/{len(todo)}] PUBLIC {name} -> {fn} ({rec['bytes']} B)", flush=True)
    else:
        rec.update({"status": f"http_{code}"})
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(todo)}] progress (last: {name} http {code})", flush=True)
    results[sid] = rec
    if (i + 1) % 25 == 0:
        json.dump(results, open("census_results.json", "w"), ensure_ascii=False, indent=1)
    time.sleep(0.4)

json.dump(results, open("census_results.json", "w"), ensure_ascii=False, indent=1)
pub = [r for r in results.values() if r.get("status") == "public"]
print(f"DONE. public sources found: {len(pub)}")
for r in pub:
    print("  PUB:", r["name"], r["file"], r["bytes"])
