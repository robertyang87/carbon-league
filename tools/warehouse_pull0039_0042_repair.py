import sys
sys.path.insert(0, '/home/hatch/workspace')
import gh_api
OWNER, REPO = 'robertyang87', 'carbon-league'
old = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/commits/e3e30d529c08ef8f6722e71d448bbafd2316b7fd')
tree = old['tree']['sha']
print('old commit tree:', tree, 'old parent:', old['parents'][0]['sha'][:12], flush=True)
ref = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/git/ref/heads/main')
head = ref['object']['sha']
print('current main:', head, flush=True)
new_commit = gh_api.req('POST', f'/repos/{OWNER}/{REPO}/git/commits',
    {'message': old['message'], 'tree': tree, 'parents': [head]})
print('new commit:', new_commit['sha'], flush=True)
patched = gh_api.req('PATCH', f'/repos/{OWNER}/{REPO}/git/refs/heads/main',
                     {'sha': new_commit['sha']})
print('ref updated to:', patched['object']['sha'], flush=True)
assert patched['object']['sha'] == new_commit['sha']
for corpus, total in [('fable6_corpus', 256), ('lite-v8_corpus', 268),
                      ('krypton-search256-v2_corpus', 105), ('krypton-cloudsearch_corpus', 125)]:
    listing = gh_api.req('GET', f'/repos/{OWNER}/{REPO}/contents/data/{corpus}?ref=main')
    n = len([e for e in listing if e['type'] == 'file'])
    print(f'readback {corpus}: {n} (expect {total})', flush=True)
    assert n == total, corpus
print('WAREHOUSE REPAIR OK', new_commit['sha'], flush=True)
