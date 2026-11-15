"""Patch the four h29 snaploops: claim poll 30s -> 90s (quota throttle).

War-room 2026-11-13 04:14: shared GitHub API pool fell 548 in 15 min
(3857/5000 left). C1's loops poll the runs list every 30s during claim;
a trip's cloud run takes ~2-4 min, so 90s polling still detects
completion promptly while cutting ~half the claim calls. Trip cadence
(INTERVAL=120) is NOT changed: stretching trips to 15 min would cut the
backfill to ~0.3 rows/min and push the remaining ~936 rows to ~24h, past
the 13:08 deadline ruling - that tradeoff is referred back to the war room
instead of being applied silently.
"""
BASE = "/home/hatch/workspace/goals/carbon-league-large-file-leaderboard-challenge/hidden_files"
FILES = ["s29-snaploop-fable.py", "s29-snaploop-fableg1.py",
         "s29-snaploop-m7v2r1.py", "s29-snaploop-m7v2g1.py"]

OLD = """    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(30)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")"""

NEW = """    deadline = time.time() + timeout_s
    while time.time() < deadline:
        # quota throttle (war-room 2026-11-13 04:14): poll every 90s, not
        # 30s - the cloud run takes minutes, so detection stays prompt
        time.sleep(90)
        try:
            r = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/actions/workflows/{WF}/runs?per_page=30")"""

for name in FILES:
    p = f"{BASE}/{name}"
    src = open(p).read()
    assert src.count(OLD) == 1, f"{name}: claim-poll pattern count={src.count(OLD)}"
    src = src.replace(OLD, NEW, 1)
    assert src.count("v3.6 linepos-window)\")") == 1, f"{name}: start-line tag not unique"
    src = src.replace("v3.6 linepos-window)\")",
                       "v3.6 linepos-window + claim90s)\")", 1)
    open(p, "w").write(src)
    print(f"patched {name}")

print("OK - claim poll 30s -> 90s on all four loops")
