"""Minimal GitHub API helper using the custom.github credential surrogate."""
import json, sys, urllib.request

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import add_surrogate_to_request

API = "https://api.github.com"

def req(method, path, body=None):
    url = API + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Accept": "application/vnd.github+json",
                                        "User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    with urllib.request.urlopen(r, timeout=30) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None

def get_file(owner, repo, path, ref="main"):
    return req("GET", f"/repos/{owner}/{repo}/contents/{path}?ref={ref}")

def put_file(owner, repo, path, content_bytes, message, sha=None, branch="main"):
    import base64
    body = {"message": message,
            "content": base64.b64encode(content_bytes).decode(),
            "branch": branch}
    if sha:
        body["sha"] = sha
    return req("PUT", f"/repos/{owner}/{repo}/contents/{path}", body)

def get_job_logs(owner, repo, job_id):
    """Fetch a job's logs as text. The API 302-redirects to a pre-signed
    Azure blob URL; following it WITH the auth header makes Azure return
    401. So: catch the redirect, fetch the blob URL with no auth header."""
    import urllib.error, urllib.parse
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{API}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    loc = None
    try:
        with opener.open(r, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    r2 = urllib.request.Request(loc, method="GET", headers={"User-Agent": "muse-agent"})
    with urllib.request.urlopen(r2, timeout=60) as resp2:
        return resp2.read().decode("utf-8", errors="replace")

def dispatch(owner, repo, workflow, ref="main", inputs=None):
    body = {"ref": ref}
    if inputs:
        body["inputs"] = inputs
    return req("POST", f"/repos/{owner}/{repo}/actions/workflows/{workflow}/dispatches", body)

if __name__ == "__main__":
    # quick self-test: repo details
    print(json.dumps(req("GET", "/repos/robertyang87/tennislive"), indent=1)[:300])
