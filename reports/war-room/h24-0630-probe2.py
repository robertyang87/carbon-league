#!/usr/bin/env python3
"""H24 0630 follow-up: decode contents-API dicts, compare vs 0614 values."""
import sys, json, base64, hashlib
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO = "robertyang87", "carbon-league"
out = {}

for label, path in [("live_wt_BrU6S", "synced/h24/bigeval_live/wt_BrU6S.jsonl"),
                    ("cal_out", "synced/h24/m3s2r1_cal.out")]:
    try:
        d = gh_api.get_file(OWNER, REPO, path, ref="sync/h24")
        text = base64.b64decode(d["content"]).decode("utf-8", "replace")
        lines = [l for l in text.splitlines() if l.strip()]
        out[label] = {"blob_sha": d.get("sha", "")[:8],
                      "size": d.get("size"),
                      "lines": len(lines),
                      "sha256": hashlib.sha256(text.encode()).hexdigest()[:8],
                      "last_line": lines[-1][:300] if lines else None}
        if label == "cal_out":
            out[label]["text"] = text
    except Exception as e:
        out[label] = {"error": str(e)[:300]}

print(json.dumps(out, indent=1, ensure_ascii=False))
