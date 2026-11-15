"""K4 probe poller: snapshot sync-branch jsonl every ~60s, wait for both fields,
then final on-box status read. Prints a full receipt at the end.
Usage: PYTHONPATH=/home/hatch/workspace timeout -k 30 900 python3 k4probe-poller.py
"""
import sys, base64, time, json, os, subprocess
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
SNAP = BASE + "/exam_snapshots/h24"
os.makedirs(SNAP, exist_ok=True)
LOGF = SNAP + "/k4probe_snapshot.log"
TAGS = ["k4v2r1_e0", "k4v2g1_e0"]
DEADLINE = 900

def fetch(path):
    try:
        r = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h24")
        if isinstance(r, dict) and r.get("content"):
            return base64.b64decode(r["content"])
        return None
    except Exception as e:
        print("fetch", path, "ERR", str(e)[:100], flush=True)
        return None

def summaries(bigeval_bytes):
    out = {}
    if not bigeval_bytes:
        return out
    for ln in bigeval_bytes.decode("utf-8", "replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if isinstance(r, dict) and r.get("tag") in TAGS:
            out[r["tag"]] = r
    return out

def status_run(marker):
    script = f"""echo {marker}_BEGIN
date "+%F %T %Z"
cat /root/carbon/data/k4v2_chain.status 2>/dev/null || echo NO-STATUS
wc -l /root/carbon/data/bigeval_live/k4v2r1_e0.jsonl /root/carbon/data/bigeval_live/k4v2g1_e0.jsonl 2>/dev/null
pgrep -fc bigeval.py || echo 0
cat /proc/loadavg
echo {marker}_END
"""
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": "i-uf68zwwjpkg8iyk2epsd", "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    t0 = time.time()
    seen = set()
    while time.time() - t0 < 300:
        time.sleep(15)
        try:
            r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=30")
        except Exception as e:
            print("list error:", e, flush=True)
            continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen or run["status"] != "completed":
                continue
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                try:
                    log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
                except Exception:
                    continue
                if isinstance(log, str) and marker in log:
                    i0 = log.find(marker + "_BEGIN")
                    i1 = log.find(marker + "_END")
                    return log[i0:i1 + len(marker) + 4] if i0 >= 0 else log[:4000]
    return "STATUS-RUN-NOT-CLAIMED"

t0 = time.time()
trips = 0
last_counts = {}
final_summ = {}
with open(LOGF, "a") as lf:
    lf.write(f"K4POLL start {time.strftime('%F %T %Z')}\n")

while time.time() - t0 < DEADLINE:
    now = time.strftime("%H:%M:%S")
    counts = {}
    for tag in TAGS:
        data = fetch(f"synced/h24/bigeval_live/{tag}.jsonl")
        if data is not None:
            n = data.count(b"\n")
            counts[tag] = n
            if last_counts.get(tag) != n:
                open(f"{SNAP}/{tag}.jsonl", "wb").write(data)
                trips += 1
                with open(LOGF, "a") as lf:
                    lf.write(f"{now} trip {trips} {tag} {n} lines\n")
                last_counts[tag] = n
    bj = fetch("synced/h24/bigeval.jsonl")
    if bj is not None:
        open(f"{SNAP}/bigeval.jsonl", "wb").write(bj)
        final_summ = summaries(bj)
    print(f"[{now}] counts={counts} summaries={ {k: ('mean', v.get('mean'), 'n', v.get('n')) for k, v in final_summ.items()} } trips={trips} elapsed={int(time.time()-t0)}s", flush=True)
    if all(counts.get(t, 0) >= 200 for t in TAGS) and len(final_summ) == 2:
        print("BOTH FIELDS COMPLETE", flush=True)
        break
    time.sleep(60)

print("--- FINAL STATUS RUN ---", flush=True)
st = status_run("K4STATFINAL")
print(st, flush=True)
open(SNAP + "/k4probe_final_status.txt", "w").write(st)

# final re-fetch of both jsonls + bigeval for the receipt
for tag in TAGS:
    data = fetch(f"synced/h24/bigeval_live/{tag}.jsonl")
    if data is not None:
        open(f"{SNAP}/{tag}.jsonl", "wb").write(data)
        print(f"final {tag}: {data.count(chr(10))} lines", flush=True)
bj = fetch("synced/h24/bigeval.jsonl")
if bj is not None:
    open(f"{SNAP}/bigeval.jsonl", "wb").write(bj)
    final_summ = summaries(bj)
print("SUMMARIES-RAW:", flush=True)
for tag in TAGS:
    print(tag, json.dumps(final_summ.get(tag), ensure_ascii=False), flush=True)
print(f"TRIPS={trips} ELAPSED={int(time.time()-t0)}", flush=True)
