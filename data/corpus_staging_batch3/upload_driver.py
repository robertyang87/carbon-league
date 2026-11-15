"""Upload staged watchlist corpora to robertyang87/carbon-league main.
Resumable: reads upload_log.md for completed paths (sha present) and skips them.
Usage: python3 upload_driver.py [max_files] [label]
Rate-limit rule: any HTTP 403 stops the run immediately (exit 2).
"""
import sys, os, time, base64, re
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
import urllib.error

STAGE = "/home/hatch/workspace/carbon-live/corpus_staging_batch3"
LOG = os.path.join(STAGE, "upload_log.md")
OWNER, REPO = "robertyang87", "carbon-league"
LABELS = ["m3s3z", "hakimi31", "k4v3", "lite-v8", "fable6", "hakimi3pro"]
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def done_paths():
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4 and parts[1].startswith("data/") and SHA_RE.match(parts[2]):
                done.add(parts[1])
    return done


def put_with_retry(lab, f, p, data):
    msg = f"data: warehouse {lab} watchlist corpus replay {f}"
    try:
        return gh_api.put_file(OWNER, REPO, p, data, msg, branch="main")
    except urllib.error.HTTPError as e:
        if e.code == 403:
            raise
        time.sleep(5)
        return gh_api.put_file(OWNER, REPO, p, data, msg, branch="main")


def main():
    max_n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    only = sys.argv[2] if len(sys.argv) > 2 else None
    done = done_paths()
    jobs = []
    for lab in LABELS:
        if only and lab != only:
            continue
        d = os.path.join(STAGE, f"{lab}_corpus")
        for f in sorted(os.listdir(d)):
            p = f"data/{lab}_corpus/{f}"
            if p not in done:
                jobs.append((lab, f, p))
    jobs = jobs[:max_n]
    print(f"jobs this run: {len(jobs)} (already done: {len(done)})", flush=True)
    t0 = time.time()
    n_ok = n_skip = n_fail = 0
    with open(LOG, "a") as log:
        for i, (lab, f, p) in enumerate(jobs, 1):
            data = open(os.path.join(STAGE, f"{lab}_corpus", f), "rb").read()
            try:
                r = put_with_retry(lab, f, p, data)
                sha = r["content"]["sha"]
                log.write(f"| {p} | {sha} | {len(data)} | {time.strftime('%H:%M:%S')} |\n")
                log.flush()
                n_ok += 1
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    print(f"RATE_LIMIT_STOP at {p}: {e}", flush=True)
                    log.write(f"| RATE_LIMIT_STOP | {p} | {time.strftime('%H:%M:%S')} |\n")
                    log.flush()
                    print(f"STOPPED: ok={n_ok} skip={n_skip} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)
                    return 2
                if e.code == 422:
                    try:
                        ex = gh_api.get_file(OWNER, REPO, p)
                        if base64.b64decode(ex["content"]) == data:
                            log.write(f"| {p} | {ex['sha']} | {len(data)} | {time.strftime('%H:%M:%S')} | already-present-identical |\n")
                            log.flush()
                            n_skip += 1
                            continue
                    except Exception:
                        pass
                    log.write(f"| {p} | CONFLICT-422 | {time.strftime('%H:%M:%S')} |\n")
                    log.flush()
                    n_fail += 1
                    continue
                log.write(f"| {p} | ERROR HTTP {e.code} | {time.strftime('%H:%M:%S')} |\n")
                log.flush()
                n_fail += 1
            except Exception as e2:
                log.write(f"| {p} | ERROR {type(e2).__name__} {str(e2)[:120]} | {time.strftime('%H:%M:%S')} |\n")
                log.flush()
                n_fail += 1
            if i % 10 == 0:
                print(f"  {i}/{len(jobs)} ok={n_ok} skip={n_skip} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)
    print(f"DONE: ok={n_ok} skip={n_skip} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


sys.exit(main())
