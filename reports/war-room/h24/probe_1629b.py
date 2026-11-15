import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo, branch = 'robertyang87', 'carbon-league', 'sync/h24'

def fetch(path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    return base64.b64decode(r['content']).decode('utf-8', errors='replace')

# cal.out full text
print('== m3s2r1_cal.out ==')
print(fetch('synced/h24/m3s2r1_cal.out'))

# E0 live file: line count, wins, last lines
txt = fetch('synced/h24/bigeval_live/m3s2r1_e0.jsonl')
lines = [l for l in txt.splitlines() if l.strip()]
print('== m3s2r1_e0.jsonl ==')
print('lines:', len(lines))
wins = 0
margins = []
for l in lines:
    try:
        d = json.loads(l)
    except Exception:
        continue
    # try common keys
    print('sample keys:', sorted(d.keys())) if lines.index(l) == 0 else None
    break
# print last 3 lines raw (truncated)
for l in lines[-3:]:
    print(l[:400])
