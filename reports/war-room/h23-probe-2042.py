"""H23 status probe 20:42 - read sync/h23 branch via GitHub API. Marker H23STAT2042K7P9."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"
print("MARKER: H23STAT2042K7P9")

def fetch_raw(path, ref):
    try:
        d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    except Exception as e:
        return f"<fetch error: {e}>"
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    return f"<no content: {str(d)[:150]}>"

# 1. latest commits (cadence check)
commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?sha={BR}&per_page=10")
print("RECENT COMMITS:")
for c in commits:
    print(" ", c["sha"][:7], c["commit"]["committer"]["date"], c["commit"]["message"][:100])

# 2. list dirs
for d in ["synced/h23", "synced/h23/bigeval_live"]:
    items = gh_api.get_file(OWNER, REPO, d, ref=BR)
    if isinstance(items, list):
        print(f"DIR {d} ({len(items)} items):")
        for it in items:
            print(f"  {it['name']}  size={it.get('size')}")
    else:
        print(f"DIR {d}: {str(items)[:200]}")

# 3. live files
for name in ["m7bsr1_m2.jsonl", "m7bsr1_echo2.jsonl", "m3s2wr1_m2.jsonl",
             "m3s2wg1_m2.jsonl", "m3s2wr1_echo2.jsonl", "m3s2wg1_echo2.jsonl",
             "m3s2g1_m2.jsonl", "m3s2g1_e0.jsonl", "echo2_m2.jsonl"]:
    raw = fetch_raw(f"synced/h23/bigeval_live/{name}", BR)
    if raw.startswith("<no content") or raw.startswith("<fetch error"):
        print(f"LIVE {name}: ABSENT ({raw[:100]})")
        continue
    lines = [l for l in raw.splitlines() if l.strip()]
    wins = 0; tot = 0.0; errs = 0; first_t = None; last_t = None
    seeds = []
    first3 = []; last3 = []
    for l in lines:
        r = json.loads(l)
        m = r.get("margin", 0)
        tot += m
        if m > 0:
            wins += 1
        if r.get("err"):
            errs += 1
        if first_t is None:
            first_t = r.get("t")
        last_t = r.get("t")
        seeds.append(r.get("seed"))
        if len(first3) < 3:
            first3.append(m)
        last3.append(m)
        last3 = last3[-3:]
    n = len(lines)
    if n:
        print(f"LIVE {name}: games={n} wins={wins} losses={n-wins} mean_margin={tot/n:.1f}",
              f"err_games={errs} first_t={first_t} last_t={last_t} seeds={seeds[0]}..{seeds[-1]} first3={first3} last3={last3}")
    else:
        print(f"LIVE {name}: games=0")

# 4. cal.out files
for name in ["m7bsr1_m2.out", "m7bsr1_echo2.out", "lineA_cal.out", "echo2_anchor.out",
             "m3s2g1_m2.out", "g1_chain.out", "m3s2wr1_m2.out", "m3s2wg1_m2.out",
             "m3s2wr1_echo2.out", "m3s2wg1_echo2.out", "e0_m2_base.out"]:
    raw = fetch_raw(f"synced/h23/{name}", BR)
    if raw.startswith("<no content") or raw.startswith("<fetch error"):
        print(f"{name}: ABSENT")
    else:
        print(f"{name}: len={len(raw)} content={raw[:800]!r}")

# 5. bigeval.jsonl tail
raw = fetch_raw("synced/h23/bigeval.jsonl", BR)
blines = [l for l in raw.splitlines() if l.strip()]
print(f"bigeval.jsonl: {len(blines)} lines")
for l in blines[-6:]:
    print("  TAIL:", l[:400])
