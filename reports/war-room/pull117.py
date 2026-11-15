import json, subprocess, sys

def pull(sid, out):
    rows = []
    off = 0
    while True:
        url = f"https://117.72.210.134/api/matches?compact=1&limit=100&offset={off}&strategy={sid}"
        r = subprocess.run(["curl", "-skL", "--max-time", "60", url], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"curl rc={r.returncode} {r.stderr[:200]}")
        d = json.loads(r.stdout)
        if isinstance(d, list):
            items = d
            nxt = None
        else:
            items = d.get("matches") or d.get("items") or d.get("data") or []
            nxt = d.get("next_offset")
        rows += items
        if not nxt:
            break
        off = nxt
    json.dump(rows, open(out, "w"), ensure_ascii=False)
    return len(rows)

JOBS = [
    ("28cd4ac93ab020aa6c1abd821675339d", "/tmp/m9xj320_117.json"),
    ("c1c405519a1a04962c2d38a343882fab", "/tmp/godslayer_cvi_117.json"),
    ("e4f8d24c6352f7b71b10e3ec16df60e9", "/tmp/carago1_117.json"),
    ("2b4b44fd148fa7fdaa53bb0d110ae705", "/tmp/fable_117.json"),
    ("42aaba6351ad55541cdf6087573d9e65", "/tmp/m9xj460_117.json"),
]
for sid, out in JOBS:
    try:
        n = pull(sid, out)
        print(out, n, flush=True)
    except Exception as e:
        print(out, "FAIL", e, flush=True)
