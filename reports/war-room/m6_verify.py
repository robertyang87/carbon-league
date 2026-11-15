"""Verify M6 corpus count in GitHub repo (source of truth) vs DGX workdir claim.
Counts data/m6_corpus/ entries and data/m6_matches.json entries on
cnet2-search-v12, and shows the latest commit touching data/m6_corpus/."""
import sys, json, base64
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

OWNER, REPO, REF = "robertyang87", "carbon-league", "cnet2-search-v12"

branch = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/branches/{REF}")
print("branch head:", branch["commit"]["sha"], branch["commit"]["commit"]["message"].splitlines()[0])

listing = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/contents/data/m6_corpus?ref={REF}")
if isinstance(listing, list):
    names = [e["name"] for e in listing]
    print("corpus entries:", len(names))
    print("sample:", names[:5])
    print("has index file:", [n for n in names if "index" in n.lower()])
else:
    print("corpus listing unexpected:", json.dumps(listing)[:300])

mf = gh_api.get_file(OWNER, REPO, "data/m6_matches.json", ref=REF)
content = base64.b64decode(mf["content"]).decode()
matches = json.loads(content)
print("matches entries:", len(matches) if isinstance(matches, list) else "not a list")
print("matches file sha:", mf.get("sha"))

commits = gh_api.req("GET", f"/repos/{OWNER}/{REPO}/commits?path=data/m6_corpus&sha={REF}&per_page=5")
for c in commits:
    print("corpus commit:", c["sha"][:12], c["commit"]["author"]["date"], c["commit"]["message"].splitlines()[0])
