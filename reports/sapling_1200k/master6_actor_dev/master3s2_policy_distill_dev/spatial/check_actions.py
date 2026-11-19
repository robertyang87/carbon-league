"""Match original submitted orders on unmodified public-entry calls from two policies using their own legal prefixes."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import numpy as np
from extract import HERE,RAW,B,F
import entry
import fit as R

def run():
    fit=json.loads((HERE/'fit.json').read_bytes())
    proto=json.loads((HERE/'protocol.json').read_bytes())
    assert fit['protocol_sha256']==R.sha(HERE/'protocol.json') and R.sha(HERE/'student.py')==fit['pack_sha256']
    packs={'raw_soft':RAW/'student.py','spatial':HERE/'student.py','parent':entry.FILES['parent']}
    assert R.sha(packs['raw_soft'])==proto['identity']['student.py']
    cards=[c for c in json.loads((RAW/'manifest.json').read_bytes())['records'] if c['split']=='held' and c['domain']=='original']
    assert len(cards)==len({c['terrain'] for c in cards})==5
    arms={k:entry.import_pack(p) for k,p in packs.items()}
    counts=[];traces=[]
    for c in cards:
        seat=c['seat']; sim,pairs=B.replay(c);fx=F.Feat(seat);summary=Counter();byrole={k:Counter() for k in ('collector','planter','base')}
        roots=[]
        for t,pair in enumerate(pairs):
            obs={k:v for k,v in sim.obs(seat).items() if k in B.PUBLIC}; x=fx.observe(obs)
            out={k:v.agent(obs,sim.config) for k,v in arms.items()}
            if t in c['steps']:
                ys={k:fx.labels(cmd) for k,cmd in out.items()};src=fx.labels(pair[seat]);prefixes={k:F.Prefix(x,fx.aux) for k in arms}
                summary['roots']+=1;summary['changed_joint']+=out['raw_soft']!=out['spatial']
                decisions=[]
                for i in np.flatnonzero(src>=0):
                    i=int(i);y=int(src[i]);leg={k:pf.feats(i)[2] for k,pf in prefixes.items()}
                    role='base' if i>=10 else 'planter' if x['tok_kind'][i]==1 else 'collector'
                    eligible=all(leg[k].sum()>1 and leg[k][y] for k in ('raw_soft','spatial'))
                    for k,pf in prefixes.items():
                        a=int(ys[k][i]);assert leg[k][a];pf.advance(i,a)
                    if eligible:
                        for r in (summary,byrole[role]):
                            r['eligible']+=1
                            for k in arms:r[k]+=int(ys[k][i])==y
                            r['changed']+=int(ys['raw_soft'][i]!=ys['spatial'][i])
                    decisions.append({'uid':fx.own_uid[i],'role':role,'logged':y,'eligible':bool(eligible),'action':{k:int(a[i]) for k,a in ys.items()},'legal':{k:np.flatnonzero(a).tolist() for k,a in leg.items()}})
                roots.append(dict(t=t,slots=decisions))
            if t==max(c['steps']):break
            sim.step(*pair)
        assert len(roots)==12 and summary['eligible']
        counts.append(dict(id=c['id'],seat=seat,terrain=c['terrain'],source_sha256=c['source_sha256'],stats=dict(summary),role={k:dict(v) for k,v in byrole.items()},delta=(summary['spatial']-summary['raw_soft'])/summary['eligible']))
        traces.append(dict(id=c['id'],terrain=c['terrain'],seat=seat,source_sha256=c['source_sha256'],roots=roots))
        print('normal-entry',c['id'],dict(summary),flush=True)
    raw=gzip.compress(json.dumps(traces,separators=(',',':'),allow_nan=False).encode(),mtime=0)
    (HERE/'action_roots.json.gz').write_bytes(raw)
    seats={s:float(np.mean([g['delta'] for g in counts if g['seat']==s])) for s in (0,1)}
    held=fit['held'];kl={k:v['all']['map_equal_kl'] for k,v in held.items()}
    macro={k:float(np.mean([g['stats'][k]/g['stats']['eligible'] for g in counts])) for k in arms}
    checks={'soft_kl_lower_than_matched':kl['spatial']<kl['raw_soft'],'true_emitted_agreement_gain':macro['spatial']>macro['raw_soft'],
            'at_least_three_positive':sum(g['delta']>0 for g in counts)>=3,'both_seats_nondecreasing':min(seats.values())>=0.}
    R.dump(HERE/'actions.json',dict(fit_identity=R.sha(HERE/'fit.json'),eval_sha256=R.sha(Path(__file__)),trace_sha256=B.sha(raw),entry_pack_sha256={k:R.sha(v) for k,v in packs.items()},checks=checks,native_response_eligible=all(checks.values()),new_maps=0,macro_agreement=macro,matched_seat_delta=seats,maps=counts,held_legal_KL=kl))
    print(macro,seats,checks,flush=True)
if __name__=='__main__':run()
