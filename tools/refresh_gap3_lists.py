import json, sys, urllib.request, ssl, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

CTX = ssl._create_unverified_context()
def fetch_live(hexid):
    out, off = [], 0
    while True:
        url = f"https://117.72.210.134/api/matches?compact=1&limit=100&offset={off}&strategy={hexid}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
            data = json.loads(r.read())
        if isinstance(data, dict):
            batch = data.get('matches') or data.get('items') or []
            nxt = data.get('next_offset')
        else:
            batch, nxt = data, None
        out.extend(batch)
        if not nxt:
            break
        off = nxt
    return out

LABELS = {
    'k4v3': ('9d596068f39e0b8cca317c3bb27f8ae5', 'data/k4v3_matches.json'),
    'fable6': ('5b453cb4f7c49439673deda697d46a9b', 'data/fable6_matches.json'),
    'hakimi3pro': ('aff6e57c5eba8857d5546783c7d39a46', 'data/hakimi3pro_matches.json'),
}
for label, (hexid, path) in LABELS.items():
    live = fetch_live(hexid)
    cur = gh_api.get_file('robertyang87', 'carbon-league', path, ref='main')
    cur_list = json.loads(base64.b64decode(cur['content']))
    print(f"[{label}] live {len(live)}, main now {len(cur_list)}", flush=True)
    if len(live) > len(cur_list):
        gh_api.put_file('robertyang87', 'carbon-league', path, json.dumps(live).encode(),
                        f"{label} matches list refresh to {len(live)} (gap-3 follow-up)", sha=cur['sha'], branch='main')
        back = gh_api.get_file('robertyang87', 'carbon-league', path, ref='main')
        back_list = json.loads(base64.b64decode(back['content']))
        print(f"[{label}] READ-BACK list {len(back_list)}", flush=True)
    else:
        print(f"[{label}] no refresh (live not ahead)", flush=True)
print("DONE", flush=True)
