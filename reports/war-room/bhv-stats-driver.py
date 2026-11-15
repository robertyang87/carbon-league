import gh_api, base64, time, os, sys
# usage: bhv-stats-driver.py <instance_id> <marker> <mode> <outlog>
INST, MARK, MODE, OUTLOG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.chdir(os.path.expanduser("~/workspace"))
base = "goals/carbon-league-large-file-leaderboard-challenge/hidden_files/"
pyb64 = base64.b64encode(open(base + "behavior_stats.py", "rb").read()).decode()
if MODE == "h24":
    tail = ('timeout 180 python3 /tmp/behavior_stats.py /root/carbon/data/bigeval_live/m3s2r1_m2.jsonl '
            '--summary-mean 1197 --summary-plants 39,11 --summary-dep 9103,9386 --summary-tree 4394,1481')
else:  # h23 pre field: find the 200-line m3s2r1pre vs Master2 file first
    tail = ('f=$(ls /root/carbon/data/bigeval_live/*m3s2r1pre*.jsonl 2>/dev/null | head -5); echo "CANDIDATES: $f"; '
            'for c in $f; do echo "$c $(wc -l < $c)"; done; '
            'tgt=""; for c in $f; do case "$c" in *e0*) continue;; esac; if [ "$(wc -l < $c)" = "200" ]; then tgt="$c"; fi; done; '
            'echo "TARGET: $tgt"; '
            'timeout 20 grep -a "m3s2r1pre" /root/carbon/data/bigeval.jsonl | tail -4 | cut -c1-600; '
            '[ -n "$tgt" ] && timeout 180 python3 /tmp/behavior_stats.py "$tgt" --summary-mean 1115 || echo NO_TARGET_FOUND')
script = (f"echo {MARK}_BEGIN\n"
          f"printf '%s' '{pyb64}' | base64 -d > /tmp/behavior_stats.py\n"
          f"python3 -c \"import ast; ast.parse(open('/tmp/behavior_stats.py').read()); print('SCRIPT_SYNTAX_OK')\"\n"
          f"{tail}\n"
          f"echo {MARK}_END\n")
t0 = time.time()
gh_api.dispatch("robertyang87", "carbon-league", "cloud-assistant.yml", ref="main",
    inputs={"instance_id": INST, "region": "cn-shanghai",
            "command_b64": base64.b64encode(script.encode()).decode()})
print("dispatched", MARK, "at", time.strftime("%H:%M:%S"), flush=True)
deadline = time.time() + 540
seen_runs = set()
result = None
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
        except Exception as e:
            print("jobs error:", e, flush=True); continue
        for j in jobs.get("jobs", []):
            try:
                log = gh_api.get_job_logs("robertyang87", "carbon-league", j["id"])
            except Exception as e:
                print("log error:", e, flush=True); continue
            if isinstance(log, str) and MARK in log:
                print("CLAIMED run", rid, "job", j["id"], "conclusion", run.get("conclusion"), flush=True)
                i0 = log.find(MARK + "_BEGIN")
                i1 = log.find(MARK + "_END")
                out = base + OUTLOG
                open(out, "w").write(log)
                print("FULL LOG SAVED", out, "chars", len(log), flush=True)
                print(log[max(0, i0 - 100):(i1 + 200 if i1 > 0 else i0 + 14000)], flush=True)
                result = log
                break
        if result:
            break
    if result:
        break
print("elapsed", round(time.time() - t0), "claimed" if result else "NOT-CLAIMED", flush=True)
