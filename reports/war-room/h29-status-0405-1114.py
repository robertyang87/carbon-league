"""War-room sync 04:05: branch heads + main recent commits + rate limit. Marker H29STATUS0405."""
import sys, json
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
OWNER, REPO = "robertyang87", "carbon-league"
def api(path): return gh_api.req("GET", path)
def bh(br):
    try:
        b = api(f"/repos/{OWNER}/{REPO}/branches/{br}")
        c = b["commit"]
        print(f"BRANCH_HEAD {br} {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:90]!r}", flush=True)
    except Exception as e:
        print(f"BRANCH_HEAD {br} ERROR {type(e).__name__}: {e}", flush=True)
for br in ["main","sync/h23","sync/h24","sync/h28","sync/h29","sync/h31"]:
    bh(br)
try:
    cs = api(f"/repos/{OWNER}/{REPO}/commits?sha=main&per_page=8")
    print("COMMITS main newest first:")
    for c in cs:
        print(f"  {c['sha'][:8]} {c['commit']['committer']['date']} {c['commit']['message'].splitlines()[0][:100]!r}", flush=True)
except Exception as e:
    print("COMMITS main ERROR", e, flush=True)
try:
    rl = api("/rate_limit")
    core = rl["resources"]["core"]
    print(f"RATE_LIMIT core remaining={core['remaining']} limit={core['limit']} reset={core['reset']}", flush=True)
except Exception as e:
    print("RATE_LIMIT ERROR", e, flush=True)
# m3s3z list head check
try:
    d = api(f"/repos/{OWNER}/{REPO}/contents/data/m3s3z_matches.json?ref=main")
    import base64
    raw = base64.b64decode(d["content"]).decode()
    ids = json.loads(raw)
    n = len(ids) if isinstance(ids, list) else -1
    print(f"M3S3Z_LIST main count={n} sha={d['sha'][:12]}", flush=True)
except Exception as e:
    print("M3S3Z_LIST ERROR", type(e).__name__, e, flush=True)
