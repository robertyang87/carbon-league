#!/usr/bin/env python3
"""Archive Echo3 bridge field to sync/h29 and verify read-back by git blob sha."""
import hashlib, json, sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h29"
SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"

def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

def put_and_verify(path, data, msg):
    r = gh_api.put_file(OWNER, REPO, path, data, msg, branch=BR)
    print("put", path, "commit", r["commit"]["sha"][:8])
    meta = gh_api.get_file(OWNER, REPO, path, ref=BR)
    ok = meta["sha"] == blob_sha(data) and meta["size"] == len(data)
    print("verify", path, "size", meta["size"], "blob_sha_match", ok)
    assert ok

a = open(f"{SNAP}/echo3bridge_a.jsonl", "rb").read()
b = open(f"{SNAP}/echo3bridge_b.jsonl", "rb").read()
put_and_verify("synced/h29/bigeval_live/echo3bridge_a.jsonl", a,
               "exam(h29): echo3bridge_a live lines (Echo3 deployment pack vs Echo2, seeds 986000-986199)")
put_and_verify("synced/h29/bigeval_live/echo3bridge_b.jsonl", b,
               "exam(h29): echo3bridge_b live lines (Echo2 vs Echo3 deployment pack, seeds 986000-986199)")

summary = json.load(open("/home/hatch/workspace/carbon-live/bridge-e3/echo3bridge-analysis.json"))
summary["final_jsonl_md5"] = dict(
    echo3bridge_a=hashlib.md5(a).hexdigest(),
    echo3bridge_b=hashlib.md5(b).hexdigest())
summary["exam_window_cst"] = "2026-11-15 00:37-00:41 CST (spot-29)"
data = json.dumps(summary, indent=1, ensure_ascii=False).encode()
put_and_verify("synced/h29/summaries/echo3bridge.json", data,
               "exam(h29): echo3bridge summary (Echo3 vs Echo2 bridge readout)")
print("ARCHIVE_OK")
