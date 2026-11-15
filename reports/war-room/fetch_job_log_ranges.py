"""Recover a GitHub Actions job log in Range chunks.

2026-11-12 23:27 probe: gh_api.get_job_logs() failed every time on a
26894-byte log - IncompleteRead(15364 bytes read, ...) plus one 503
'Egress is over the account limit'. The egress path truncates responses
at ~15KB, so a plain full-log fetch can never succeed on logs bigger
than that. This helper fetches the same 302 -> Azure blob URL as
gh_api.get_job_logs (no auth header on the blob request) but in 8KB
Range chunks with retries, then concatenates.

Usage: python3 fetch_job_log_ranges.py <job_id> [out_path]
Prints whether the caller-supplied marker (env MARK) is present.
"""
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/home/hatch/workspace")
from gh_api import API  # noqa: E402
from dynamic_credentials import add_surrogate_to_request  # noqa: E402


def fetch_log(job_id, chunk=8000, attempts=3):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    url = f"{API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET",
                               headers={"User-Agent": "muse-agent"})
    add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")
        if not loc:
            raise
    # Total size unknown; probe in chunk-sized ranges until a short read.
    parts = []
    start = 0
    while True:
        data = None
        for attempt in range(attempts):
            try:
                req2 = urllib.request.Request(
                    loc, method="GET",
                    headers={"User-Agent": "muse-agent",
                             "Range": f"bytes={start}-{start + chunk - 1}"})
                with urllib.request.urlopen(req2, timeout=60) as resp2:
                    data = resp2.read()
                break
            except Exception as exc:  # noqa: BLE001 - retry then surface
                print(f"range {start}-{start + chunk - 1} attempt {attempt}: "
                      f"{exc!r}", file=sys.stderr)
                time.sleep(8)
        if data is None:
            raise RuntimeError(f"range starting {start} failed after retries")
        parts.append(data)
        if len(data) < chunk:
            break
        start += chunk
    return b"".join(parts).decode("utf-8", errors="replace")


if __name__ == "__main__":
    job_id = sys.argv[1]
    log = fetch_log(job_id)
    print("log chars:", len(log))
    mark = os.environ.get("MARK", "")
    if mark:
        print("marker present:", mark in log)
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(log)
        print("wrote", sys.argv[2])
