"""Check a previously frozen enemy-return predictor on the actual midgame opportunities."""
import sys
from coverage import HERE,sr
sys.path.insert(0,str(HERE.parent/'freight_horizon'))
from predictor import FreightPredictor
def run():
    model=FreightPredictor()
    simple=FreightPredictor(HERE.parent/'freight_horizon/baseline.json')
    rows=[]
    for case in sr.load(HERE/'coverage.json.gz'):
        roots={r['t']:r for r in case['roots']};history=[]
        raw=sr.load(sr.ROOT/case['source']);seat=case['seat']
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=case['seed'])
        for t,cmd in enumerate(raw['actions'][:201]):
            obs=sim.obs(seat)
            if t in roots:
                r=roots[t];uid=r['target'];hit=bool(r['target_banks_next10'])
                p=model.predict(obs,history)[uid]['p_deposit_10'];b=simple.predict(obs,history)[uid]['p_deposit_10']
                rows.append(dict(seed=case['seed'],seat=seat,t=t,uid=uid,p=p,b=b,hit=hit))
            history=(history+[obs])[-5:]
            sim.step(*cmd)
    sr.retain(HERE/'signal.json',rows)
    for key in ('p','b'):
        print(key,sum((r[key]-r['hit'])**2 for r in rows)/len(rows))
        for cutoff in (0,.65):
            q=[r for r in rows if r[key]>=cutoff]
            print(cutoff,len(q),sum(r['hit'] for r in q))
if __name__=='__main__':run()
