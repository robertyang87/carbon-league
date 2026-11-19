from fit import HERE,ACTOR,sr
from probe import signed_branch
import sys
sys.path.insert(0,str(ACTOR/'two_turn_intercept_dev'))
from ledger import check

def run():
    findings=[]
    parity=sr.load(HERE/'parity.json')
    for s in (1,0):
        out=HERE/f'seat{s}';p=sr.load(out/'live_protocol.json');raw=sr.load(sr.ROOT/p['record']['source'])
        assert any(r['source']==p['record']['source'] and p['root_t'] in r['changed'] for r in parity['sources'])
        rows=[]
        for arm in ('parent','approach'):
            rec=check(out,s,arm);cash=signed_branch(out,arm,p,raw)
            rows.append(dict(arm=arm,cash=cash,ledger=rec))
        a,b=rows
        findings.append(dict(seed=p['seed'],seat=s,t=p['root_t'],rows=rows,economy=sum(b['cash'].values())-sum(a['cash'].values()),margin=b['ledger']['margin']-a['ledger']['margin']))
    eligible=all(r['economy']>1e-8 for r in findings) and sum(r['margin'] for r in findings)>0
    sr.retain(HERE/'gate.json',dict(protocol=sr.sha(HERE/'protocol.json'),pack=parity['pack'],roots=findings,eligible=eligible))
    return eligible
if __name__=='__main__':print(run())
