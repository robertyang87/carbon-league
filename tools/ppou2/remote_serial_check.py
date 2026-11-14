"""Serial-driver state read (runs ON L20). One JSON object per arm, one line each.

Fields: arm, warm (warmup lines), formal (non-warm lines), max_it (highest it
seen), last_ckpt_sha (sha256 recorded in the newest train line), state_sha
(sha256 of state_last.pt on disk), state_match (recorded == on-disk).
"""
import hashlib, json, os

for a in ("m", "c"):
    d = f"/root/carbon/data/rl3/ppou2_{a}"
    tp = os.path.join(d, "train.jsonl")
    warm = formal = 0
    max_it = 0
    last_sha = None
    if os.path.exists(tp):
        for line in open(tp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            it = r.get("it", 0)
            if it > max_it:
                max_it = it
            if r.get("warm"):
                warm += 1
            else:
                formal += 1
                last_sha = (r.get("checkpoint") or {}).get("files", {}).get("state_last.pt")
    sp = os.path.join(d, "state_last.pt")
    state_sha = None
    if os.path.exists(sp):
        h = hashlib.sha256()
        with open(sp, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        state_sha = h.hexdigest()
    print(json.dumps({"arm": a, "warm": warm, "formal": formal, "max_it": max_it,
                      "last_ckpt_sha": last_sha, "state_sha": state_sha,
                      "state_match": bool(last_sha and state_sha and last_sha == state_sha)}))
