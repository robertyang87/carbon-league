"""Fetch a GitHub Actions job log via curl (urllib blob reads currently die
with IncompleteRead/RemoteDisconnected through the egress proxy; curl --http1.1 works)."""
import sys, urllib.request, urllib.error, subprocess, os
sys.path.insert(0, "/home/hatch/workspace")
sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request
API = "https://api.github.com"

def get_location(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return None  # no redirect: tiny log inline (rare)
    except urllib.error.HTTPError as e:
        return e.headers.get("Location")

def fetch(job_id, out_path, tries=3):
    loc = get_location(job_id)
    if not loc:
        return None
    for i in range(tries):
        p = subprocess.run(["curl", "-sS", "--http1.1", "--max-time", "60", "-o", out_path, "-w", "%{http_code}:%{size_download}", loc],
                           capture_output=True, text=True)
        ok = p.stdout.strip()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0 and ok.startswith("200:"):
            return out_path
        print(f"  curl try {i} -> {ok} err={p.stderr.strip()[:100]}", flush=True)
    return None

if __name__ == "__main__":
    job_id, out_path = sys.argv[1], sys.argv[2]
    r = fetch(job_id, out_path)
    print("OK" if r else "FAIL", out_path, os.path.getsize(out_path) if r else "")
