import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
def api(path): return gh_api.req("GET", path)
# summaries dir listing
summ = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/summaries?ref=sync/h29")
print(f"SUMMARIES entries={len(summ)}: {[e['name'] for e in summ]}")
# live files via contents API (base64)
files = ["fableg1_echo2.jsonl","fableg1_m2.jsonl","fabler1_echo2.jsonl","fabler1_m2.jsonl",
         "m7bsv2g1_echo2.jsonl","m7bsv2g1_m2.jsonl","m7bsv2r1_echo2.jsonl","m7bsv2r1_m2.jsonl"]
total = 0
for name in files:
    f = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/bigeval_live/{name}?ref=sync/h29")
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    seeds = []
    last_seed = last_ts = None
    for ln in lines:
        try:
            seeds.append(json.loads(ln).get("seed"))
        except Exception:
            pass
    if lines:
        try:
            last = json.loads(lines[-1])
            last_seed = last.get("seed")
            last_ts = last.get("ts") or last.get("time") or last.get("timestamp") or last.get("ended_at")
        except Exception:
            last_seed = lines[-1][:100]
    print(f"  {name}: lines={len(lines)} unique_seeds={len(set(seeds))} last_seed={last_seed} last_ts={last_ts}")
    total += len(lines)
print(f"TOTAL live lines = {total} / 1600")
rl = api("/rate_limit"); print("RATE remaining:", rl["resources"]["core"]["remaining"])
