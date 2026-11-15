"""H23 status 12:12 follow-up verify (run 2026-11-14 ~12:14 CST).
Verify a4g1 both fields truly complete: unique seeds, wins, mean margin,
err count, worst_ms from the live lines (cal.out not yet in sync/h29).
List synced/h29/summaries. Marker: H23VERIFY1212.
"""
import sys, json, base64, statistics
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
HF = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
LOG = f"{HF}/h23-verify-1212-1114.log"

def api(path):
    return gh_api.req("GET", path)

out = []
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)

p("MARKER H23VERIFY1212")

try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/summaries?ref=sync/h29")
    names = sorted(x["name"] for x in d) if isinstance(d, list) else [str(d)]
    p("DIR synced/h29/summaries: " + ", ".join(names))
except Exception as e:
    p(f"DIR synced/h29/summaries ERROR {type(e).__name__}: {e}")

for name in ["a4g1_a4_m2.jsonl", "a4g1_base_m2.jsonl"]:
    try:
        f = api(f"/repos/{OWNER}/{REPO}/contents/synced/h29/bigeval_live/{name}?ref=sync/h29")
        raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
        rows = []
        for ln in raw.splitlines():
            if ln.strip():
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
        seeds = [r.get("seed") for r in rows]
        margins = [r.get("margin") for r in rows if isinstance(r.get("margin"), (int, float))]
        wins = sum(1 for r in rows if r.get("win") is True)
        errs = sum(1 for r in rows if r.get("err"))
        wms = [r.get("wms") for r in rows if isinstance(r.get("wms"), (int, float))]
        ts = [r.get("t") for r in rows if r.get("t")]
        p(f"VERIFY {name}: rows={len(rows)} unique_seeds={len(set(seeds))} "
          f"seed_range={min(seeds)}-{max(seeds)} wins={wins} "
          f"mean_margin={statistics.fmean(margins):.1f} "
          f"err_rows={errs} max_wms={max(wms) if wms else None} "
          f"t_range={min(ts)}..{max(ts)}")
    except Exception as e:
        p(f"VERIFY {name} ERROR {type(e).__name__}: {e}")

try:
    b = api(f"/repos/{OWNER}/{REPO}/branches/sync/h29")
    c = b["commit"]
    p(f"BRANCH_HEAD sync/h29 {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}")
except Exception as e:
    p(f"BRANCH_HEAD sync/h29 ERROR {type(e).__name__}: {e}")

open(LOG, "w").write("\n".join(out) + "\n")
p("saved", LOG)
