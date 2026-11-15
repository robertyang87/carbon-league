import urllib.error, urllib.request
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"

def fetch_log_tolerant(job_id):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/{OWNER}/{REPO}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            data = resp.read()
            return data.decode("utf-8", errors="replace"), "full"
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    try:
        with urllib.request.urlopen(r2, timeout=60) as resp2:
            data = resp2.read()
            return data.decode("utf-8", errors="replace"), "full"
    except Exception as e:
        partial = getattr(e, "partial", b"") or b""
        return partial.decode("utf-8", errors="replace"), f"partial:{type(e).__name__}"
