"""Read all four receipts and independently check intervention scope, doses and terminal score pairs."""
import gzip
import json

from register import HERE,IDS,GUARD,require,sha,save


def verify():
    plan=json.loads((HERE/'protocol.json').read_bytes())
    batches=json.loads((HERE/'results.json').read_bytes())
    require(len(batches['results'])==4 and batches['protocol_sha256']==sha((HERE/'protocol.json').read_bytes()),'Incomplete batch')
    paired=[]
    for spec in plan['streams']:
        branches={}
        for result in [r for r in batches['results'] if r['stream']==spec['stream']]:
            raw=(HERE/result['artifact']).read_bytes()
            require(sha(raw)==result['sha256'],'Mismatched receipt')
            b=json.loads(gzip.decompress(raw)); rows=b['rows'];branches[result['arm']]=b
            require(rows[0]['step']==286 and rows[-1]['step']==298 and len(rows)==13,'Missing actions')
            require(result['scores']==rows[-1]['cash'] and result['scores']==result['cash'],'Terminal score and cash differ')
            for r in rows:
                require(r['issued'][1]==r['commands'][1] and r['clocks'][0]['wall']>0 and r['clocks'][1]['wall']>0,'Opponent command or timer')
                own,actual=r['commands'][0],r['issued'][0]
                changed={k for k in (set(own)|set(actual)) if own.get(k)!=actual.get(k)}
                require(changed==set() or (result['arm']=='intercept_return' and changed=={IDS[0]}),'Changed another worker')
            for s,uid in ((0,IDS[0]),(1,IDS[1]),(1,GUARD)):
                ds=[(r['step'],e[2]) for r in rows for e in r['cash_events'] if e[0]=='own_delivery' and e[1]==s and e[3]==uid and e[2]>0]
                require([list(x) for x in ds]==result['workers'][uid]['deposits'],'Deposit attribution')
        native,alt=[branches[k] for k in ('natural','intercept_return')]
        require(native['root']==alt['root'] and native['rows'][0]['commands']==alt['rows'][0]['commands'],'Not a common simultaneous root')
        a,b=native['result'],alt['result']
        paired.append(dict(stream=spec['stream'],baseline_margin=a['margin'],modified_margin=b['margin'],margin_change=round(b['margin']-a['margin'],2),
                           own_cash_change=round(b['cash'][0]-a['cash'][0],2),enemy_cash_change=round(b['cash'][1]-a['cash'][1],2),
                           natural_own_deposits=a['workers'][IDS[0]]['deposits'],treatment_own_deposits=b['workers'][IDS[0]]['deposits']))
    return {'pairs':paired,'moves':sum(r['moves'] for r in batches['results']),'one_physical_map':True}
if __name__=='__main__':print(save(HERE/'audit.json',verify()))
