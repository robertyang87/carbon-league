"""Factual direction versus immediate purchase/next-step successful siting, with map-level averaging."""
from collections import Counter,defaultdict
from math import log
from statistics import mean
from sources import HERE,read,save,sha
from model import OUTCOMES

def stats(data, prior):
    count=Counter(); fraction=defaultdict(list)
    for r in data:
        a=r['a'];y=r['target'];stage='execute' if a==0 else 'site_move';label=OUTCOMES[y]
        count['n']+=1;count[stage]+=1;count[stage+'/'+label]+=1
        for labelhead in ('native_only','primary'):
            raw,selected,p,s,pa=r[labelhead]
            for variant,pick in (('BC',raw),('economic',selected)):
                k=f'{labelhead}/{variant}/'
                correct=pick==a
                for group in ('all',stage,stage+'/'+label):
                    count[k+group]+=correct
                if variant=='economic':
                    count[k+'differs_from_bc']+=pick!=raw
                    count[k+'differs_from_demonstrator']+=not correct
                fraction[k+'exact'].append(float(correct))
            k=labelhead+'/'
            fraction[k+'action_log_loss'].append(-log(max(1e-12,p[a])))
            fraction[k+'economic_log_loss'].append(-log(max(1e-12,s[y])))
            fraction[k+stage+'/productive_brier'].append((s[2]-int(y==2))**2)
            fraction[k+stage+'/p_productive'].append(s[2])
            fraction[k+stage+'/truth_productive'].append(float(y==2))
        fraction['naive/'+stage+'/productive_brier'].append((prior[stage]-int(y==2))**2)
    return dict(count=dict(count),rates={k:mean(v) for k,v in fraction.items()})

def main():
    records=read(HERE/'sources.json')['records']; rows=read(HERE/'decisions.json.gz')
    output={}
    for domain in ('native','elite'):
        fit=[r for r in rows if records[r['game']]['split']=='fit' and records[r['game']]['domain']==domain]
        prior={name:mean(r['target']==2 for r in fit if (r['a']==0)==stay)
               for name,stay in [('execute',True),('site_move',False)]}
        z={}
        for split in ('fit','held','exposed_loss_panel'):
            rs=[r for r in rows if records[r['game']]['split']==split and records[r['game']]['domain']==domain]
            if not rs:continue
            aggregate=stats(rs,prior)
            maps={m:stats([r for r in rs if records[r['game']]['terrain']==m],prior)
                  for m in sorted({records[r['game']]['terrain'] for r in rs})}
            groups={}
            for f in ('teacher_result','actor','seat'):
                groups[f]={k:stats([r for r in rs if str(records[r['game']][f])==k],prior) for k in sorted({str(records[r['game']][f]) for r in rs})}
            keys=aggregate['rates']
            z[split]=dict(pooled=aggregate,by_map=maps,by_group=groups,
                          equal_map_rates={k:mean(m['rates'][k] for m in maps.values() if k in m['rates']) for k in keys})
        output[domain]=dict(fit_productive_rate=prior,panels=z)
    save(HERE/'evaluation.json',dict(domains=output,scorer_sha256=sha(HERE/'evaluate.py'),
          decisions_sha256=sha(HERE/'decisions.json.gz'),raw_sources_sha256=sha(HERE/'sources.json')))
if __name__=='__main__':main()
