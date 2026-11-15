import sys, hashlib, base64, os
sys.path.insert(0, "/home/hatch/workspace")
import gh_api

O, R = "robertyang87", "carbon-league"
head = gh_api.req("GET", f"/repos/{O}/{R}/branches/main")
print("main head:", head["commit"]["sha"], head["commit"]["commit"]["committer"]["date"])

listing = gh_api.req("GET", f"/repos/{O}/{R}/contents/code/exp?ref=main")
print("exp entries:", len(listing))
by_name = {}
for e in listing:
    by_name[e["name"]] = e
    low = e["name"].lower()
    if any(k in low for k in ("it85", "m3s2", "snipe")):
        print("MATCH", e["name"], e["size"], e["sha"])

e = by_name.get("it85.py")
assert e, "it85.py missing from code/exp"
blob = gh_api.req("GET", f"/repos/{O}/{R}/git/blobs/{e['sha']}")
data = base64.b64decode(blob["content"])
print("it85.py bytes:", len(data), "md5:", hashlib.md5(data).hexdigest())
os.makedirs("/home/hatch/workspace/carbon-live/it85", exist_ok=True)
with open("/home/hatch/workspace/carbon-live/it85/it85.py", "wb") as f:
    f.write(data)

m = by_name.get("m3s2r1.py")
if m:
    print("repo m3s2r1.py size:", m["size"], "blob sha:", m["sha"])
    fobj = gh_api.get_file(O, R, "code/exp/m3s2r1.py", ref="main")
    txt = base64.b64decode(fobj["content"]).decode("utf-8", "replace")
    print("--- repo code/exp/m3s2r1.py ---")
    print(txt)
else:
    print("no m3s2r1.py in repo code/exp")
