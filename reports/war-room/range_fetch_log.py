"""Fetch a GitHub Actions job log in byte ranges (Azure blob supports Range).
Usage: python3 range_fetch_log.py <job_id> [start] [end]
Prints the fetched slice to stdout; exit 0 on success."""
import sys, urllib.request, urllib.error
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

job_id = sys.argv[1]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end = sys.argv[3] if len(sys.argv) > 3 else ""

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

opener = urllib.request.build_opener(NoRedirect)
url = f"{gh_api.API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
try:
    with opener.open(r, timeout=30) as resp:
        data = resp.read()
        sys.stdout.buffer.write(data)
        sys.exit(0)
except urllib.error.HTTPError as e:
    loc = e.headers.get("Location")
    if not loc:
        raise
rng = f"bytes={start}-" + (end if end else "")
r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent", "Range": rng})
with urllib.request.urlopen(r2, timeout=60) as resp2:
    sys.stdout.buffer.write(resp2.read())
