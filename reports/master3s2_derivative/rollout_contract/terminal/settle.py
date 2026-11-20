from collections import Counter,defaultdict
from pathlib import Path
import gzip,json
from register import HERE,fp

def main():
    raw=(HERE/'protocol.json').read_bytes();p=fp(raw);plan=json.loads(raw);rows=[];maps=defaultdict(list)
    for t in plan['tickets']:
        for i,ds in enumerate(t['streams']):
            arms={};raws={}
            for n in ['stock','home']:
                file=HERE/f"{t['seed']}_{t['step']}_{i}_{n}.json.gz"
                j=json.loads(gzip.decompress(file.read_bytes()));r=j['rows']
                assert j['plan']==p and j['stream']==ds and j['t']==t['step'] and j['status']==['DONE','DONE'] and j['seed']==t['seed']
                assert j['final_step']==299 and [x['step'] for x in r]==list(range(t['step'],299))
                assert r[0]['actions'][1]==t['orders'][n]
                assert r[0]['actions'][0]==j['opponent_root']==(j['source_reference_root'][0] if i==0 else j['native_root'][0])
                assert j['margin']==round(j['official_scores'][1]-j['official_scores'][0],2)
                econ=Counter()
                for x in r:
                    for k,s,v,*_ in x['events']:econ[k]+=v*(1 if s==1 else -1)
                assert abs((r[-1]['cash'][1]-r[-1]['cash'][0])-j['margin'])<1e-6
                times=[c['wall'] for x in r for c in x['clocks']]
                arms[n]=dict(sha256=fp(file.read_bytes()),file=file.name,margin=j['margin'],scores=j['official_scores'],cashflow=dict(econ),slow=sum(x>1 for x in times),max_wall=max(times),steps=len(r))
                raws[n]=j
            assert raws['stock']['root_private']==raws['home']['root_private'] and raws['stock']['opponent_root']==raws['home']['opponent_root'] and raws['stock']['rows'][0]['public']==raws['home']['rows'][0]['public']
            change=round(arms['home']['margin']-arms['stock']['margin'],2)
            if i:maps[t['seed']].append(change)
            d=Counter(arms['home']['cashflow']);d.subtract(arms['stock']['cashflow'])
            assert abs(sum(d.values())-change)<1e-6
            opp=next((v['step'] for v,w in zip(raws['stock']['rows'],raws['home']['rows'],strict=True) if v['actions'][0]!=w['actions'][0]),None)
            rows.append(dict(seed=t['seed'],seat=1,t=t['step'],stream=ds['kind'],delta=change,cash_diff_components=dict(d),opponent_diverges=opp,arms=arms))
    av={k:sum(v)/len(v) for k,v in maps.items()};summary=dict(plan=p,pairs=rows,map_mean_fresh=av,map_equal_mean_fresh=sum(av.values())/len(av))
    (HERE/'settlement.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    for r in rows:
        d=r['cash_diff_components'];print(r['seed'],r['t'],r['stream'],r['delta'],r['opponent_diverges'], sorted(((round(x,2),k) for k,x in d.items()),key=lambda y:-abs(y[0]))[:3])
    print(summary['map_mean_fresh'],summary['map_equal_mean_fresh'])
if __name__=='__main__':main()
