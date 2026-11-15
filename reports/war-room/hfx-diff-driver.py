import gh_api, base64, time, os, sys
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
tmpl = open(base + "hfx-diff.sh").read()
targets = {"H23": ("i-uf65ykinp4dvkirzk7q4", "HFXH23A7"),
           "H24": ("i-uf68zwwjpkg8iyk2epsd", "HFXH24B8"),
           "S25": ("i-uf65ykinp4dw4551xjci", "HFXS25C9")}
marks = {}
for name, (iid, mark) in targets.items():
    script = tmpl.replace("__MARK__", mark)
    gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
        inputs={"instance_id": iid, "region": "cn-shanghai",
                "command_b64": base64.b64encode(script.encode()).decode()})
    marks[mark] = name
    print("dispatched", name, mark, time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 600
seen_runs = set(); found = {}
while time.time() < deadline and len(found) < 3:
    time.sleep(20)
    try:
        r = gh_api.req("GET", "/repos/robertyang87/carbon-league/actions/workflows/cloud-assistant.yml/runs?per_page=15")
    except Exception as e:
        print("list error:", e, flush=True); continue
    for run in r.get("workflow_runs", [])[:15]:
        rid = run["id"]
        if rid in seen_runs or run["status"] != "completed": continue
        seen_runs.add(rid)
        try:
            jobs = gh_api.req("GET", f"/repos/robertyang87/carbon-league/actions/runs/{rid}/jobs")
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if not isinstance(log, str): continue
            for mark, name in marks.items():
                if mark in log and name not in found:
                    i0 = log.find(mark + "_BEGIN"); i1 = log.find(mark + "_END")
                    seg = log[i0:i1 + len(mark) + 4] if i0 >= 0 and i1 > i0 else log[log.find(mark)-100:log.find(mark)+2000]
                    # strip GH log timestamps
                    clean = "\n".join(l.split("Z ", 1)[-1] if "Z " in l[:40] else l for l in seg.splitlines())
                    found[name] = clean
                    print(f"===== {name} ({mark}) run {rid} =====\n{clean}", flush=True)
print("RESULT", {k: ("OK" if k in found else "MISSING") for k in targets}, flush=True)
