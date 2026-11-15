import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api, urllib.error, urllib.request
MARK = "STUDIOSTAT0112"
JOBS = [
    (43267223126, 131353940176),
    (43267435595, 131354710871),
    (43267434620, 131354702061),
    (43267433584, 131354701390),
    (43267433421, 131354704113),
    (43267790785, 131355940777),
    (43267894568, 131356307823),
]

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

def get_location(job_id):
    opener = urllib.request.build_opener(NoRedirect)
    url = f"{gh_api.API}/repos/robertyang87/carbon-league/actions/jobs/{job_id}/logs"
    r = urllib.request.Request(url, method="GET", headers={"User-Agent": "muse-agent"})
    gh_api.add_surrogate_to_request(r, "custom.github", allowed_hosts=["api.github.com"])
    try:
        with opener.open(r, timeout=30) as resp:
            return None  # no redirect: full log inline (unlikely)
    except urllib.error.HTTPError as e:
        return e.headers.get("Location")
    except Exception as e:
        print("loc error", job_id, e, flush=True)
        return None

for rid, jid in JOBS:
    loc = get_location(jid)
    if not loc:
        print(rid, jid, "no-location", flush=True)
        continue
    for rng in ("bytes=15000-60000", "bytes=0-60000"):
        r2 = urllib.request.Request(loc, method="GET",
            headers={"User-Agent": "muse-agent", "Range": rng})
        try:
            with urllib.request.urlopen(r2, timeout=60) as resp2:
                data = resp2.read()
            txt = data.decode("utf-8", errors="replace")
            print(rid, jid, rng, "got", len(data), "marker" if MARK in txt else "no-marker", flush=True)
            if MARK in txt:
                i0 = txt.find(MARK)
                print(txt[max(0, i0-200):i0+16000], flush=True)
                sys.exit(0)
            break  # range worked, no need for second range on this job
        except Exception as e:
            partial = getattr(e, "partial", b"") or b""
            txt = partial.decode("utf-8", errors="replace")
            print(rid, jid, rng, "err", type(e).__name__, "partial", len(partial),
                  "marker" if MARK in txt else "no-marker", flush=True)
            if MARK in txt:
                i0 = txt.find(MARK)
                print(txt[max(0, i0-200):i0+16000], flush=True)
                sys.exit(0)
print("RANGE-PROBE DONE: no marker in any job", flush=True)
