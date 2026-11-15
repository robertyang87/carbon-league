import sys, json, base64
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

owner, repo = 'robertyang87', 'carbon-league'

def head(branch):
    commits = gh_api.req('GET', f'/repos/{owner}/{repo}/commits?sha={branch}&per_page=1')
    c = commits[0]
    return c['sha'][:8], c['commit']['committer']['date'], c['commit']['message'].splitlines()[0][:120]

def listdir(branch, path):
    r = gh_api.get_file(owner, repo, path, ref=branch)
    if isinstance(r, list):
        return [(d['name'], d['size']) for d in r]
    return [('NOT_A_DIR', str(r)[:200])]

for b in ['sync/h24', 'sync/h31', 'sync/h29', 'main']:
    try:
        sha, dt, msg = head(b)
        print(f'{b}: {sha} @ {dt} :: {msg}')
    except Exception as e:
        print(f'{b}: HEAD ERROR {e}')

for b, path in [('sync/h24', 'synced/h24'), ('sync/h31', 'synced/h31'),
                ('sync/h29', 'synced/h29')]:
    try:
        items = listdir(b, path)
        outs = [n for n, s in items if n.endswith('.out')]
        print(f'{b} {path}: {len(items)} items; .out={outs}')
        live = listdir(b, path + '/bigeval_live')
        print(f'{b} {path}/bigeval_live: {len(live)} files')
    except Exception as e:
        print(f'{b} listing ERROR {e}')
