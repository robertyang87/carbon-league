"""H23 status probe 23:12 - small machine direct read (marker H23STATUS2312).
Read-only: chain status, live-file line counts, bigeval procs, loadavg,
launcher-log tails. No b64 dumps (avoids the log-truncation issue that
breaks the snapshot loop's B64 extraction). Claims by unique marker only.
"""
import base64, json, sys, time

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, WF = "robertyang87", "carbon-league", "cloud-assistant.yml"
L20 = "i-uf62k0o6zbe6j7zxnzg5"
MARK = "H23STATUS2312"
LOGF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/h23-probe-2312-driver.log"

def log(s):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}"
    print(line, flush=True)
    with open(LOGF, "a") as f:
        f.write(line + "\n")

SCRIPT = f"""MARK={MARK}
echo "$MARK L20-BEGIN $(date '+%F %T %Z')"
timeout -k 30 300 ssh -i /root/.ssh/dgx_access -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new arm@10.211.43.155 'bash -s' <<'DGXEOF'
MARK={MARK}
TGT=47.117.170.99
KEY=/home/arm/.ssh/id_ed25519_carbon_relay
echo "$MARK DGX-BEGIN"
timeout -k 30 180 ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 root@"$TGT" 'bash -s' <<'TGTEOF'
echo "TGT-BEGIN $(date '+%F %T %Z')"
echo "--- CHAIN-STATUS ---"
cat /root/carbon/data/fabler1_cal.status 2>/dev/null || echo NO-STATUS-FILE
echo "--- LIVE-COUNTS ---"
wc -l /root/carbon/data/bigeval_live/fabler1_m2.jsonl /root/carbon/data/bigeval_live/fabler1_echo2.jsonl 2>&1
echo "--- BIGEVAL-PROCS ---"
ps -eo pid,etime,args | grep '[b]igeval.py' || echo NO-BIGEVAL-PROC
echo "--- LOAD ---"
cat /proc/loadavg; uptime
echo "--- LAUNCHER-TAILS ---"
tail -n 3 /root/carbon/data/fabler1_m2.launcher.log 2>/dev/null | cut -c1-200
echo "-- echo2 launcher --"
tail -n 3 /root/carbon/data/fabler1_echo2.launcher.log 2>/dev/null | cut -c1-200 || echo NO-ECHO2-LAUNCHER-LOG
echo "--- SUMMARIES-ON-DISK ---"
grep -o '"tag": "fabler1[^"]*"' /root/carbon/data/bigeval.jsonl 2>/dev/null || echo NO-SUMMARIES
echo "TGT-END"
TGTEOF
echo "TGT-RC=$?"
echo "$MARK DGX-END"
DGXEOF
echo "$MARK L20-END rc=$?"
"""

def main():
    log(f"dispatch marker={MARK}")
    gh_api.dispatch(OWNER, REPO, WF, ref="main",
                    inputs={"instance_id": L20, "region": "cn-shanghai",
                            "command_b64": base64.b64encode(SCRIPT.encode()).decode()})
    log("dispatched, claiming by marker (per_page=30 scan)")
    seen = set()
    deadline = time.time() + 480
    while time.time() < deadline:
        time.sleep(20)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")
        except Exception as e:
            log(f"claim list error: {e}")
            continue
        for run in r.get("workflow_runs", [])[:30]:
            rid = run["id"]
            if rid in seen or run["status"] != "completed":
                continue
            seen.add(rid)
            try:
                jobs = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/runs/{rid}/jobs")
            except Exception:
                continue
            for j in jobs.get("jobs", []):
                try:
                    lg = gh_api.get_job_logs(OWNER, REPO, j["id"])
                except Exception:
                    continue
                if isinstance(lg, str) and MARK in lg:
                    log(f"CLAIMED run={rid} job={j['id']}")
                    i = lg.find(MARK)
                    print("===== CLAIMED LOG (marker onward) =====")
                    print(lg[i:i + 6000])
                    print("===== END =====")
                    with open(LOGF, "a") as f:
                        f.write(lg[i:i + 6000] + "\n")
                    return
    log("NOT-CLAIMED within 480s")

if __name__ == "__main__":
    main()
