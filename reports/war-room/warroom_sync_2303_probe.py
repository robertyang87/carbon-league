import sys
sys.path.insert(0, "/home/hatch/workspace")
import gh_api
O, R = "robertyang87", "carbon-league"
for br in ["main", "sync/h23", "sync/h24", "sync/h28", "sync/h29", "sync/h31"]:
    try:
        b = gh_api.req("GET", f"/repos/{O}/{R}/branches/{br}")
        c = b["commit"]["commit"]
        print(br, b["commit"]["sha"][:8], c["committer"]["date"])
    except Exception as e:
        print(br, "ERR", e)
try:
    r = gh_api.req("GET", "/rate_limit")
    print("core remaining:", r["resources"]["core"]["remaining"], "reset:", r["resources"]["core"]["reset"])
except Exception as e:
    print("rate ERR", e)
