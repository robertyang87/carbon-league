import gh_api
for d in ["code", "reports", "tools"]:
    try:
        r = gh_api.req("GET", f"/repos/robertyang87/carbon-league/contents/{d}?ref=main&per_page=100")
        names = [x["name"] for x in r] if isinstance(r, list) else r
        print("DIR", d, ":", names, flush=True)
    except Exception as e:
        print("DIR FAIL", d, type(e).__name__, e, flush=True)
try:
    r = gh_api.req("GET", "/search/code?q=repo:robertyang87/carbon-league+TTA")
    print("SEARCH TTA total:", r.get("total_count"), [i["path"] for i in r.get("items", [])][:20], flush=True)
except Exception as e:
    print("SEARCH FAIL", type(e).__name__, e, flush=True)
