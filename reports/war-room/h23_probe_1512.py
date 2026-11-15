import sys, json
sys.path.insert(0, '/home/hatch/workspace')
import gh_api

def show(label, fn):
    try:
        r = fn()
        print(f'--- {label} ---')
        print(json.dumps(r, ensure_ascii=False)[:4000])
    except Exception as e:
        print(f'--- {label} ERROR: {e!r} ---')

show('branch sync/h23', lambda: gh_api.req('GET', '/repos/robertyang87/carbon-league/branches/sync/h23'))
show('branch main head', lambda: gh_api.req('GET', '/repos/robertyang87/carbon-league/branches/main'))
show('synced/h23 dir', lambda: gh_api.req('GET', '/repos/robertyang87/carbon-league/contents/synced/h23?ref=sync/h23'))
show('bigeval_live dir', lambda: gh_api.req('GET', '/repos/robertyang87/carbon-league/contents/synced/h23/bigeval_live?ref=sync/h23'))
