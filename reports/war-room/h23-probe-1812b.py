"""H23 status probe 18:12 part 2 - live files + cal.out + bigeval.jsonl tail."""
import json, sys, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h23"

def fetch_raw(path, ref):
    d = gh_api.get_file(OWNER, REPO, path, ref=ref)
    if isinstance(d, dict) and d.get("content"):
        return base64.b64decode(d["content"]).decode("utf-8", errors="replace")
    if isinstance(d, dict):
        return f"<no content: {d.get('message', d)[:120]}>"
    return f"<unexpected: {str(d)[:120]}>"

# 1. list bigeval_live
items = gh_api.get_file(OWNER, REPO, "synced/h23/bigeval_live", ref=BR)
print("BIGEVAL_LIVE:")
for it in items:
    print(f"  {it['name']}  size={it.get('size')}")

# 2. current live file e0_m2_base.jsonl
for name in ["e0_m2_base.jsonl"]:
    raw = fetch_raw(f"synced/h23/bigeval_live/{name}", BR)
    lines = [l for l in raw.splitlines() if l.strip()]
    print(f"\n{name}: {len(lines)} lines")
    if lines:
        first = json.loads(lines[0])
        last = json.loads(lines[-1])
        print("  first:", json.dumps(first, ensure_ascii=False)[:300])
        print("  last :", json.dumps(last, ensure_ascii=False)[:300])
        # aggregate wins/mean over games array if present
        wins = 0; tot = 0; n = 0; errs = 0
        for l in lines:
            try:
                r = json.loads(l)
            except Exception:
                continue
            for g in r.get("games", []):
                n += 1
                if g[1] == 1:
                    wins += 1
                tot += g[2]
            errs += r.get("err", 0)
        if n:
            print(f"  AGG games={n} wins={wins} losses={n-wins} mean_margin={tot/n:.1f} err_sum={errs}")

# 3. e0_m2_base.out
raw = fetch_raw("synced/h23/e0_m2_base.out", BR)
print(f"\ne0_m2_base.out: len={len(raw)} content={raw[:500]!r}")

# 4. bigeval.jsonl tail: last summary lines
raw = fetch_raw("synced/h23/bigeval.jsonl", BR)
lines = [l for l in raw.splitlines() if l.strip()]
print(f"\nbigeval.jsonl: {len(lines)} lines")
for l in lines[-6:]:
    print("  TAIL:", l[:400])
