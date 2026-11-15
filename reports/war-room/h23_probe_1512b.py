import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

# full recursive tree of sync/h23 head (tree sha from branch read)
tree = gh_api.req('GET', '/repos/robertyang87/carbon-league/git/trees/d34b91629fe8dbdc36c4819cc69dbff554975ac8?recursive=1')
paths = [t['path'] for t in tree.get('tree', [])]
h23 = [p for p in paths if p.startswith('synced/h23/')]
live = [p for p in paths if p.startswith('synced/h23/bigeval_live/')]
print('synced/h23 total items:', len(h23))
print('bigeval_live items:', len(live))
print('cal-ish files:', [p for p in h23 if p.endswith('.out')])
print('tree truncated:', tree.get('truncated'))
for p in [x for x in h23 if x.endswith('.out')]:
    f = gh_api.get_file('robertyang87', 'carbon-league', p, ref='sync/h23')
    content = base64.b64decode(f['content']).decode('utf-8', 'replace')
    print(f'--- {p} (sha {f["sha"][:8]}, {f["size"]}B) ---')
    print(content)
# rate limit
rl = gh_api.req('GET', '/rate_limit')
print('rate remaining:', rl['resources']['core']['remaining'], 'reset:', rl['resources']['core']['reset'])
