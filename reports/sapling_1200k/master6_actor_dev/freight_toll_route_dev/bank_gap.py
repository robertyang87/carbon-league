"""Separate own banking and opponent suppression, decomposing deposit volume by number and size."""
from collections import defaultdict
from ladder import collect,sr,HERE

def run():
    names=['claude-AlphaCarbon-Master6','claude-AlphaCarbon-Master5s']
    groups=defaultdict(list)
    for r in collect(include_stages=True):
        if r['player'] in names and r['opponent'] not in names:
            groups[r['opponent'],r['seat'],r['player']].append(r)
    cases=[]
    for opponent,seat in sorted({(k[0],k[1]) for k in groups}):
        arms=[groups[opponent,seat,name] for name in names]
        if min(map(len,arms))<2:continue
        for lo in (0,100,200):
            for side in (0,1):
                halves=[]
                for arm in arms:
                    vals=[r['stages'].get(lo,[0,0,0,0]) for r in arm]
                    bank=sum(v[side] for v in vals)/len(vals)
                    count=sum(v[2+side] for v in vals)/len(vals)
                    halves.append((bank,count,bank/count if count else 0))
                (b1,c1,v1),(b0,c0,v0)=halves
                frequency=(c1-c0)*(v1+v0)/2
                size=(v1-v0)*(c1+c0)/2
                assert abs((b1-b0)-frequency-size)<1.e-6
                cases.append(dict(opponent=opponent,seat=seat,side=side,start=lo,
                        games=list(map(len,arms)),banks=halves,cash_change=b1-b0,frequency_contribution=frequency,
                        mean_size_contribution=size))
    out=[]
    for lo in (0,100,200):
        for side in (0,1):
            s=[r for r in cases if r['start']==lo and r['side']==side]
            def equal_opponents(k):
                grouped=defaultdict(list)
                for r in s:grouped[r['opponent']].append(r[k])
                return sum(sum(v)/len(v) for v in grouped.values())/len(grouped)
            out.append(dict(start=lo,side=side,**{k:equal_opponents(k) for k in ('cash_change','frequency_contribution','mean_size_contribution')}))
    sr.retain(HERE/'bank_components.json',dict(players=names,definition='Master6 minus Master5s; side 0 is registered player, side 1 same opponent; extinct games contribute zero for missing late stages.',by_stratum=cases,summary=out))
    print(out)
if __name__=='__main__':run()
