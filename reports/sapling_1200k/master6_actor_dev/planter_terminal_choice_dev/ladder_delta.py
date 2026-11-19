"""Recover newly indexed completed games and quantify both sides' income without training on them."""
from concurrent.futures import ThreadPoolExecutor
import gzip,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'all_top5'))
import fetch
A='53061cb8f'
B='8658ba4c15f86bc3a528257b69379f70142d5c29'

def cards(rev,p):
    data=fetch.git('show',rev+':'+p)
    doc=json.loads(gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data)
    return {fetch.key_of(c):c for c in fetch.walk_matches(doc) if c.get('state','done')=='done' and {c['name_a'],c['name_b']}&fetch.NAMES.keys()}

def recover(row):
    key=fetch.key_of(row);p=fetch.CACHE/'replays'/(key+'.json.gz')
    url=fetch.SERVER+'/replay/'+row['id']+'/'+str(row['game'])
    data=p.read_bytes() if p.exists() else fetch.Retrieval({},0).get(url)
    env,_,_=fetch.extract(data); verified=fetch.validate(env,row)
    canonical=fetch.canonical(env)
    if not p.exists():
        tmp=p.with_suffix('.tmp');tmp.write_bytes(gzip.compress(canonical,mtime=0));tmp.replace(p)
    first=env['steps'][0][0]['observation']
    cash0=[a[0] for a in first['players']]
    paid=[0.,0.];rooted=[0,0];captured=[0,0];old={};largest=[]
    for i,pair in enumerate(env['steps']):
        ob=pair[0]['observation']
        current={tid:s for s,player in enumerate(ob['players']) for tid in player[3]}
        for tid,s in current.items():
            paid[s]+=ob['trees'][tid][1]
            if tid not in old:rooted[s]+=1
            elif old[tid]!=s:captured[s]+=1
        old=current
        if i>=10:
            before=env['steps'][i-10][0]['observation']['players']
            swing=(ob['players'][0][0]-ob['players'][1][0])-(before[0][0]-before[1][0])
            largest.append((abs(swing),swing,i))
    _,swing,end=max(largest,default=(0,0,None))
    return dict(id=key,names=[row['name_'+k] for k in row['order']],scores=env['rewards'],
                verified=verified,terrain=fetch.sha(bytes(first['full_carbon'])),canonical_sha256=fetch.sha(canonical),
                tree_payments=paid,planted=rooted,captured=captured,maximum_ten_turn_swing_seat0=swing,end=end)

if __name__=='__main__':
    paths=[p for p in fetch.git('diff','--name-only',A,B,'--','data').decode().splitlines() if fetch.sourcefile(p)]
    prior={};now={}
    for p in paths:
        prior.update(cards(A,p));now.update(cards(B,p))
    keys=sorted(now.keys()-prior.keys())
    with ThreadPoolExecutor(2) as pool:res=list(pool.map(recover,(now[k] for k in keys)))
    (HERE/'ladder_delta.json').write_bytes(fetch.canonical(dict(base=A,end=B,indices=paths,games=res)))
    print(len(res),[(x['names'],x['scores']) for x in res])
