"""Push the pool-discrimination probe summary to synced/h29/summaries/probe.json
and verify by readback."""
import hashlib
import json
import os
import sys

sys.path.insert(0, "/home/hatch/workspace")
import gh_api

SNAP = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files/exam_snapshots/h29"
du = json.load(open(os.path.join(SNAP, "probe-du.json")))
arch = json.load(open(os.path.join(SNAP, "probe-archive.json")))

summary = {
    "probe": "pool_discrimination_it95_vs_m3s2g1",
    "purpose": "budget final-review input only; not a checkpoint, no R3 consumed, §17 baseline untouched",
    "executor": "C1 / spot-29",
    "exam_window_cst": "2026-11-14 16:17:26-16:21:27",
    "seeds": "990000-990199 (probe domain; exam blocks 996000/999000 untouched)",
    "harness": {"bigeval.py": "354bcd1bc2dc8bc188c813f53d28c893", "eval_v53.py": "45792dfbea3101812836fe77e6752706"},
    "candidate": {"name": "it95 frozen net (snipe_n1 it95)", "package": "exp/it95.py", "md5": "9e16085acd3549df210c6209ab3fd3cb"},
    "baseline": {"name": "m3s2r1 pre-U frozen", "package": "exp/m3s2r1_base.py", "md5": "11d8005930b298290c2e150949349647"},
    "opponent": {"name": "m3s2g1", "package": "exp/m3s2g1_base.py", "blob_sha": "79514aeb0c835f3038520ce52baae262f3f3935c", "bytes": 3728418, "md5": "bb997d7137395e6e66f7b03134d96897"},
    "guardrail": "field 1 used the it95 frozen package; it96 and later states forbidden",
    "smoke": {
        "probe_smoke": {"wins": 0, "mean": -767, "err": 0, "FAIL": False, "md5": "c716580f6b30050e195538bfb8bf1896", "archive_commit": "aa20d591"},
        "probebase_smoke": {"wins": 2, "mean": 820, "err": 0, "FAIL": False, "md5": "3452d3d9245a19e269dbebb83759103e", "archive_commit": "3128a10e"},
    },
    "field1_it95_vs_m3s2g1": {
        "tag": "probe_it95_g1", "wins": du["candidate_wins"], "mean_margin": du["candidate_mean_margin"],
        "mean_u": du["candidate_mean_u"], "err": 0, "worst_ms": 123.0, "FAIL": False,
        "md5": du["candidate_md5"], "archive_commit": "69c5585f", "archive_sha": arch["probe_it95_g1"]["archive_sha"],
    },
    "field2_baseline_vs_m3s2g1": {
        "tag": "probe_base_g1", "wins": du["baseline_wins"], "mean_margin": du["baseline_mean_margin"],
        "mean_u": du["baseline_mean_u"], "err": 0, "worst_ms": 123.0, "FAIL": False,
        "md5": du["baseline_md5"], "archive_commit": "509718ec", "archive_sha": arch["probe_base_g1"]["archive_sha"],
    },
    "paired": {
        "n_pairs": du["n_pairs"], "dU": du["dU"], "dU_se_paired": du["dU_se_paired"],
        "ci95_normal": du["ci95_normal"], "ci95_cluster_bootstrap": du["ci95_cluster_bootstrap"],
        "bootstrap_B": du["bootstrap_B"], "p_boot_le_0": du["p_boot_le_0"],
        "pairs_pos_neg_zero": du["pairs_pos_neg_zero"],
    },
    "integrity": {
        "seed_sets_exact": "990000-990199 each once per field",
        "err_rows": 0,
        "draw_rows": [{"tag": "probe_base_g1", "seed": 990111, "margin": 0.0, "u": 0.5}],
        "max_u_recompute_dev": du["max_u_recompute_dev"],
        "ah_bh_hh_single_and_on_value": True,
        "md5_three_way_machine_local_archive": True,
    },
    "certify": "not invoked; no certificate issued",
    "archive_branch": "sync/h29 (ops-room confirmed 2026-11-14 16:19 CST; dispatch text sync/h31 was a copy error from the it95 Echo2 field)",
}
raw = json.dumps(summary, indent=2).encode()
path = "synced/h29/summaries/probe.json"
try:
    cur = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
    sha = cur["sha"]
except Exception:
    sha = None
r = gh_api.put_file("robertyang87", "carbon-league", path, raw, "probe summary it95 vs m3s2g1 (pool discrimination)", sha=sha, branch="sync/h29")
commit = r["commit"]["sha"]
print("summary push commit:", commit)
back = gh_api.get_file("robertyang87", "carbon-league", path, ref="sync/h29")
import base64
back_raw = base64.b64decode("".join(back["content"].split()))
print("readback md5:", hashlib.md5(back_raw).hexdigest(), "local md5:", hashlib.md5(raw).hexdigest())
assert hashlib.md5(back_raw).hexdigest() == hashlib.md5(raw).hexdigest()
print("SUMMARY PUSH OK commit", commit[:8])
with open(os.path.join(SNAP, "probe-summary.json"), "w") as f:
    json.dump({**summary, "summary_commit": commit}, f, indent=2)
