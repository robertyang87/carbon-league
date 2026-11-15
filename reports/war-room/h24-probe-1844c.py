import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
owner, repo = "robertyang87", "carbon-league"
branch = "sync/h24"

def read(path):
    f = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(f["content"]).decode("utf-8", "replace")

# E0 field live
raw = read("synced/h24/bigeval_live/m3s2r1_e0.jsonl")
lines = [l for l in raw.strip().splitlines() if l.strip()]
print("m3s2r1_e0.jsonl lines:", len(lines))
wins = 0; tot = 0.0; err = 0; last_t = None
for l in lines:
    d = json.loads(l)
    m = d.get("margin", 0) or 0
    tot += m
    if m > 0: wins += 1
    if d.get("err"): err += 1
    last_t = d.get("t")
print(f"wins={wins}/{len(lines)} mean={tot/len(lines):.1f} err={err} last_t={last_t}")
print("last line:", lines[-1][:400])
# last 15 games
tail = [json.loads(l) for l in lines[-15:]]
tw = sum(1 for d in tail if (d.get("margin") or 0) > 0)
tm = sum(d.get("margin") or 0 for d in tail)/len(tail)
print(f"last15: wins={tw}/15 mean={tm:.1f}")

# cal.out (Master2 field result)
print("\n== m3s2r1_cal.out ==")
print(read("synced/h24/m3s2r1_cal.out")[:3000])

# echo2 smoke (unchanged check)
raw2 = read("synced/h24/bigeval_live/echo2_smoke.jsonl")
l2 = [l for l in raw2.strip().splitlines() if l.strip()]
print("\necho2_smoke.jsonl lines:", len(l2), "last:", l2[-1][:200] if l2 else None)
