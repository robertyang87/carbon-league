"""Carbon League public-source census, phase 1+2: enumerate (id, name) pairs.

Sources: lb_live.json + replays/leaderboard.json (ratings), replays/index.json,
replays/accumulated/index.json (names only), /api/matches paging (id+name).
Output: census_ids.json
"""
import json
import subprocess
import time

BASE = "https://132.232.176.153"
PAGES = 80  # 80 x 100 = up to 8000 matches

id2name = {}
ratings = {}  # id -> rating
names_only = set()


def add(sid, name):
    if sid and name:
        id2name[sid] = name


# --- leaderboards (ratings + ids) ---
for fn in ("lb_live.json", "replays/leaderboard.json"):
    try:
        lb = json.load(open(fn))
    except Exception as e:
        print(fn, "skip:", e)
        continue
    items = lb if isinstance(lb, list) else lb.get("strategies", lb.get("items", []))
    for it in items:
        add(it.get("id"), it.get("name"))
        if it.get("id") and it.get("rating") is not None:
            ratings[it["id"]] = it["rating"]
    print(fn, "cumulative ids:", len(id2name))

# --- replay snapshot index (ids + names) ---
idx = json.load(open("replays/index.json"))
for m in idx:
    for s in ("a", "b"):
        st = m["strategies"][s]
        add(st["id"], st["name"])
print("after replays/index:", len(id2name))

# --- accumulated index (names only) ---
acc = json.load(open("replays/accumulated/index.json"))
for v in acc.values():
    names_only.add(v["a"])
    names_only.add(v["b"])
print("accumulated names:", len(names_only))

# --- matches API paging (ids + names) ---
new_per_page = []
for p in range(PAGES):
    off = p * 100
    url = f"{BASE}/api/matches?limit=100&offset={off}"
    try:
        out = subprocess.run(
            ["curl", "-sk", "--max-time", "60", url],
            capture_output=True, text=True, timeout=90,
        )
        d = json.loads(out.stdout)
    except Exception as e:
        print(f"page {p} (offset {off}) failed: {e}")
        break
    ms = d.get("matches", [])
    if not ms:
        print(f"page {p}: empty, stop")
        break
    before = len(id2name)
    for m in ms:
        add(m.get("a"), m.get("name_a"))
        add(m.get("b"), m.get("name_b"))
    new_per_page.append(len(id2name) - before)
    if p % 10 == 0 or p == PAGES - 1:
        print(f"page {p}: +{len(id2name) - before} new, total {len(id2name)}")
    time.sleep(0.5)

name2ids = {}
for sid, nm in id2name.items():
    name2ids.setdefault(nm, []).append(sid)
unresolved_names = sorted(n for n in names_only if n not in name2ids)
print("unique ids:", len(id2name))
print("accumulated names unresolved to an id:", len(unresolved_names))

json.dump(
    {
        "id2name": id2name,
        "ratings": ratings,
        "names_only_unresolved": unresolved_names,
        "new_per_page": new_per_page,
    },
    open("census_ids.json", "w"),
    ensure_ascii=False,
    indent=1,
)
print("saved census_ids.json")
