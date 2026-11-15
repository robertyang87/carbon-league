import sys, base64, json, math, statistics
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"
BR = "sync/h29"

summaries = {
    "synced/h29/summaries/it85_m2.json": {"tag": "it85_m2", "a": "exp/it85.py", "b": "exp/opp_master2r.py", "seed0": 996000, "n": 200, "err": 0, "wins": 191, "mean": 1158, "se": 57, "lo2se": 1044, "worst": -1558, "worst_ms": 120.0, "dep": [8948, 9327], "tree": [4358, 1516], "deathcargo": [197, 257], "plants": [38, 12], "t": "07:15", "search_games": 0, "changed0_games": 0, "a_md5": "6bd5cc5aa85ca90cb5426b545b7a9d84", "b_md5": "2aebfbc1008d3475ece8e178696e292d", "harness_md5": "dceb1ebeb2a804068c2263fc9206833a", "search_error_games": 0, "a_errors_total": 0, "root_errors_total": 0, "first_error": None, "first_search_error_seed": None, "FAIL": False},
    "synced/h29/summaries/it85base_m2.json": {"tag": "it85base_m2", "a": "exp/m3s2r1.py", "b": "exp/opp_master2r.py", "seed0": 996000, "n": 200, "err": 0, "wins": 181, "mean": 1182, "se": 62, "lo2se": 1057, "worst": -926, "worst_ms": 137.0, "dep": [9082, 9377], "tree": [4387, 1478], "deathcargo": [215, 215], "plants": [39, 11], "t": "07:16", "search_games": 0, "changed0_games": 0, "a_md5": "73542cdd0ea50dce87e3638fd4c9bf5a", "b_md5": "2aebfbc1008d3475ece8e178696e292d", "harness_md5": "dceb1ebeb2a804068c2263fc9206833a", "search_error_games": 0, "a_errors_total": 0, "root_errors_total": 0, "first_error": None, "first_search_error_seed": None, "FAIL": False},
}
for path, rec in summaries.items():
    sha = None
    try:
        sha = gh_api.get_file(O, R, path, ref=BR)["sha"]
        print(path, "already exists, sha", sha[:8])
    except Exception:
        pass
    if sha is None:
        out = gh_api.put_file(O, R, path, (json.dumps(rec, indent=2) + "\n").encode(),
                              f"h29 summary {rec['tag']}", branch=BR)
        print(path, "committed:", out["commit"]["sha"][:8])

def read_branch(path):
    f = gh_api.get_file(O, R, path, ref=BR)
    return base64.b64decode(f["content"]).decode(), f["sha"]

for tag in ("it85_m2", "it85base_m2"):
    txt, sha = read_branch(f"synced/h29/bigeval_live/{tag}.jsonl")
    rows = [json.loads(l) for l in txt.splitlines()]
    margins = [r["margin"] for r in rows]
    wins = sum(1 for r in rows if r["win"])
    mean = statistics.mean(margins)
    se1 = statistics.stdev(margins) / math.sqrt(len(rows)) if len(rows) > 1 else 0.0
    errs = sum(1 for r in rows if r["err"] is not None)
    seeds = sorted(r["seed"] for r in rows)
    hh = set(r["hh"][:8] for r in rows)
    ah = set(r["ah"][:8] for r in rows)
    us = [r["u"] for r in rows]
    print(f"{tag}: branch_sha={sha[:8]} lines={len(rows)} wins={wins} mean={mean:.2f} se={se1:.2f} "
          f"err={errs} seeds_ok={seeds == list(range(996000, 996200))} hh={hh} ah={ah} "
          f"u_all_float={all(isinstance(x, float) for x in us)} umean={statistics.mean(us):.4f}")

# pairing check: same seed -> same side in both fields
t1, _ = read_branch("synced/h29/bigeval_live/it85_m2.jsonl")
t2, _ = read_branch("synced/h29/bigeval_live/it85base_m2.jsonl")
s1 = {json.loads(l)["seed"]: json.loads(l)["side"] for l in t1.splitlines()}
s2 = {json.loads(l)["seed"]: json.loads(l)["side"] for l in t2.splitlines()}
print("pairing side match all seeds:", s1 == s2)
# descriptive paired dU (not the official ruling)
u1 = {json.loads(l)["seed"]: json.loads(l)["u"] for l in t1.splitlines()}
u2 = {json.loads(l)["seed"]: json.loads(l)["u"] for l in t2.splitlines()}
du = [u1[s] - u2[s] for s in sorted(u1)]
print(f"paired dU mean (descriptive only) = {statistics.mean(du):+.4f}")
