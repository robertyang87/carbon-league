"""h24 status probe, run 2026-11-14 ~16:00 CST. Marker: h24-1600."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, BR = "robertyang87", "carbon-league", "sync/h24"

# 1. Branch head
br = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{BR}")
print("BRANCH head sha:", br["commit"]["sha"])
print("BRANCH head date:", br["commit"]["commit"]["committer"]["date"])
print("BRANCH head msg:", br["commit"]["commit"]["message"][:120])

# 2. Top-level synced/h24 listing
top = gh_api.get_file(OWNER, REPO, "synced/h24", ref=BR)
print("TOP count:", len(top))
for e in top:
    if e["name"].endswith("_cal.out") or e["name"].endswith(".md"):
        print("TOP item:", e["name"], e["size"], e["sha"])

# 3. bigeval_live listing
live = gh_api.get_file(OWNER, REPO, "synced/h24/bigeval_live", ref=BR)
print("LIVE count:", len(live))
names = sorted(e["name"] for e in live)
print("LIVE sample tail:", names[-5:])

# 4. cal.out full text (only the known one unless new ones appear)
cals = [e for e in top if e["name"].endswith("_cal.out")]
for e in cals:
    f = gh_api.get_file(OWNER, REPO, "synced/h24/" + e["name"], ref=BR)
    raw = base64.b64decode(f["content"]).decode("utf-8", errors="replace")
    print("=== CAL", e["name"], "sha", e["sha"], "size", e["size"], "===")
    print(raw)

# 5. rate limit
rl = gh_api.req("GET", "/rate_limit")
print("RATE core remaining:", rl["resources"]["core"]["remaining"], "/", rl["resources"]["core"]["limit"])
