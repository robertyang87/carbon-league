import json, time, subprocess

def get(path):
    out = subprocess.run(["curl","-sk","--max-time","30","https://132.232.176.153"+path], capture_output=True, text=True)
    if out.returncode!=0:
        raise RuntimeError(f"curl failed {out.returncode} {out.stderr[:200]}")
    return json.loads(out.stdout)

def fetch_all(sid, fname):
    out=[]; offset=0
    while True:
        d=get(f"/api/matches?compact=1&limit=100&offset={offset}&strategy={sid}")
        ms=d.get("matches",[])
        out.extend(ms)
        print(fname, "offset", offset, "got", len(ms), "next", d.get("next_offset"), flush=True)
        if d.get("next_offset") is None: break
        offset=d["next_offset"]
        time.sleep(0.7)
    seen=set(); uniq=[]
    for m in out:
        k=(m["id"], m["game"])
        if k in seen: continue
        seen.add(k); uniq.append(m)
    json.dump(uniq, open(fname,"w"), indent=1)
    print("total unique", len(uniq))
    return uniq

fetch_all("8ba98152ec16fd3141276264b73755c6","matches_alpha2.json")
fetch_all("0042b70d28aced7b4305c035deaa018c","matches_alpha1.json")
