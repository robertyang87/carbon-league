"""Recheck actual original scores, timings, complete registration and published evidence without issuing decisions."""
import hashlib
import json
from pathlib import Path
from native import HERE,OUT,PLAN,REG,read,sha,save,require,driver,check,summarize

def verify():
    binding=read(OUT/'binding.json')
    protocol=read(PLAN)
    require(sha(PLAN)==binding['registered_sha256']==REG and binding['candidate']==protocol['candidate']['sha256'],'candidate or registration')
    for name,expected in binding['checked_driver_sha256'].items():
        require(sha(HERE.parents[3]/name)==expected,'changed driver '+name)
    results=[]
    artifacts=[PLAN,HERE/'native.py',HERE/'report.py',Path(__file__),HERE/'clearance.json',OUT/'binding.json']
    for i,s in enumerate(protocol['seeds']):
        for seat in protocol['seats']:
            j=('candidate',s,seat,(i+seat)%2,binding['boards'][str(s)],binding)
            result=check(j)
            d=OUT/'candidate'
            artifacts.extend(d/f'{name}_{s}_{seat}.{ext}' for name,ext in
                             [('raw','json.gz'),('ticket','json'),('row','json'),('receipt','json'),('wall','json.gz')])
            results.append(result)
    require(len(results)==16 and json.loads(json.dumps(summarize(results)))==read(HERE/'comparison.json'),'signed results or clock differ')
    if any(r['result']=='win' for r in results):
        first=read(HERE/'first_win.json')
        require(first['game'] in results and first['game']['result']=='win' and 1<=first['signed_completed_when_reported']<=16,'first win source')
        artifacts.append(HERE/'first_win.json')
    else:
        require(not (HERE/'first_win.json').exists(),'unexpected first-win artifact')
    artifacts.extend(HERE/k for k in ('comparison.json','runtime.json','RESULTS.md'))
    indexed={str(f.relative_to(HERE)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(artifacts)}
    signature=hashlib.sha256(''.join(k+'\t'+v['sha256']+'\n' for k,v in indexed.items()).encode()).hexdigest()
    receipt={'files':indexed,'artifacts':len(artifacts),'total_bytes':sum(x['bytes'] for x in indexed.values()),
             'index_digest':signature,'official_games_replayed':len(results),'complete_wins':sum(r['result']=='win' for r in results)}
    save(HERE/'verification.json',receipt)
    print(signature,len(artifacts),receipt['complete_wins'],flush=True)
if __name__=='__main__':verify()
