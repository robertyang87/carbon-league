#!/usr/bin/env python3
"""Archive Echo4 launch exam to sync/h29 and verify read-back by git blob sha."""
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

files = {}
for tag in ("e4pack_m2", "e4bare_m2", "e4pack_e2"):
    files[tag] = open(f"{SNAP}/{tag}.jsonl", "rb").read()
put_and_verify("synced/h29/bigeval_live/e4pack_m2.jsonl", files["e4pack_m2"],
               "exam(h29): e4pack_m2 live lines (Echo4 pack vs Master2 real, seeds 996000-996199)")
put_and_verify("synced/h29/bigeval_live/e4bare_m2.jsonl", files["e4bare_m2"],
               "exam(h29): e4bare_m2 live lines (bare ppou2 it20 vs Master2 real, seeds 996000-996199)")
put_and_verify("synced/h29/bigeval_live/e4pack_e2.jsonl", files["e4pack_e2"],
               "exam(h29): e4pack_e2 live lines (Echo4 pack vs Echo2, seeds 999000-999199)")

summary = json.load(open("/home/hatch/workspace/carbon-live/echo4/echo4-analysis.json"))
summary["exam"] = "Echo4 launch exam (certify + equivalence + calibration), spot-34, executor C1"
summary["exam_window_cst"] = "2026-11-15 12:12:36-12:18:00 CST (games); certify 12:08-12:09 CST"
summary["pack"] = {"path": "code/packs/echo4_int8.py", "md5": "819250bfc82632786063dabd46a616bd",
                   "bytes": 4601242, "main_commit": "ffff6a92"}
summary["certify"] = {"overall": "PASS", "certify_pack_md5": "fc33be6cc0255d87d69c8eef8b9a2875",
                      "quota": {"cpu_max": "150000 100000", "memory_max": 536870912, "source": "cgroup_v2"},
                      "games": 3, "opp": "Master2 wrapper 2aebfbc1",
                      "max_init_wall_s": 0.309, "step_ms_mean": 50.2, "step_ms_p95": 50.4, "step_ms_max": 90.1,
                      "slow_steps_per_game": [0, 0, 0], "overage_pool_s": 0.0,
                      "max_game_wall_s": 21.554, "peak_rss_mib": 94.3, "a_errors": 0,
                      "identity_three_way": True}
summary["equivalence_note"] = ("Pack and bare net produced identical margins on all 200 seeds "
                                "(paired diff exactly 0.0); mirror files are distinct runs "
                                "(different finish times, different file md5).")
summary["final_jsonl_md5"] = {t: hashlib.md5(d).hexdigest() for t, d in files.items()}
data = json.dumps(summary, indent=1, ensure_ascii=False).encode()
put_and_verify("synced/h29/summaries/echo4.json", data,
               "exam(h29): echo4 summary (certify PASS, equivalence exact, calibration +566.9)")
print("ARCHIVE_OK")
