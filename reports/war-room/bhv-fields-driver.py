import gh_api, base64, time, os
MARK = "BHVTREE1801Z3"
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
pysrc = '''
import json
from statistics import mean
rows = [json.loads(l) for l in open("/root/carbon/data/bigeval_live/m3s2r1_m2.jsonl") if l.strip()]
def means(getter, keys):
    out = {}
    for k in keys:
        vals = []
        for r in rows:
            d = getter(r)
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        if len(vals) >= 100:
            out[k] = round(mean(vals), 1)
    return out
polkeys = set()
for r in rows:
    polkeys.update((r.get("lx") or {}).get("pol", {}).keys())
mp = means(lambda r: (r.get("lx") or {}).get("pol", {}), sorted(polkeys))
mo = means(lambda r: (r.get("lx") or {}).get("opp", {}), sorted(polkeys))
print("POL_MEANS", json.dumps(mp))
print("OPP_MEANS", json.dumps(mo))
ta = [ (r["lx"]["pol"]["treesteps"] + r["lx"]["pol"]["colsteps"]) for r in rows ]
tb = [ (r["lx"]["opp"]["treesteps"] + r["lx"]["opp"]["colsteps"]) for r in rows ]
print("TREESTEPS_PLUS_COLSTEPS_AB", round(mean(ta),1), round(mean(tb),1))
'''
script = (f"echo {MARK}_BEGIN\n"
          f"printf '%s' '{base64.b64encode(pysrc.encode()).decode()}' | base64 -d > /tmp/bhv_fields.py\n"
          f"timeout 120 python3 /tmp/bhv_fields.py\n"
          f"echo {MARK}_END\n")
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": "i-uf68zwwjpkg8iyk2epsd", "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 480
seen_runs = set(); result = None
while time.time() < deadline:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=12")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:12]:
        rid = run["id"]
        if rid in seen_runs or run["status"] != "completed":
            continue
        seen_runs.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception:
            continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception:
                continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK + "_BEGIN"); i1 = log.find(MARK + "_END")
                open(base + "bhv-fields-h24-full.log", "w").write(log)
                print(log[max(0, i0 - 100):(i1 + 200 if i1 > 0 else i0 + 12000)], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
