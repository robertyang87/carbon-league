import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api
for p in ['synced/h23/lineA_cal.out', 'synced/h23/m3s2r1pre_cal.out', 'synced/h23/m7bsr1_cal.out']:
    f = gh_api.get_file('robertyang87', 'carbon-league', p, ref='sync/h23')
    content = base64.b64decode(f['content']).decode('utf-8', 'replace')
    print(f'--- {p} (sha {f["sha"][:8]}, {f["size"]}B) ---')
    print(content)
rl = gh_api.req('GET', '/rate_limit')
print('rate remaining:', rl['resources']['core']['remaining'], 'reset epoch:', rl['resources']['core']['reset'])
