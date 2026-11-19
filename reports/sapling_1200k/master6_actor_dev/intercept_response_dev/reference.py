"""Compare the two frozen forecast heads on identical original-opponent moves."""
import math
from collections import deque
from match import HERE,ACTOR,native
from forecast import carriers,_DIRECTION
import numpy as np

def main():
    model=native.module(ACTOR/'opponent_cargo_calibration_dev/model.py','prior_freight_forecast')
    prior=model.Forecast()
    assert list(model.intent.MOVES)==['NONE',*_DIRECTION[1:]]
    w=np.load(HERE/'weights.npz')['weights'];rows=[]
    for folder in ('planter_escape_full_dev','planter_payback_full_dev','planter_capture_full_dev','capture_reach_dev'):
        for path in sorted((ACTOR/folder/'games').glob('*/raw_*.json.gz')):
            raw=native.read(path);s=raw['candidate_seat'];history=deque(maxlen=11)
            sim=native.exam.FastCarbon(native.exam.CFG).reset(seed=raw['seed'])
            sums=np.zeros(4);n=0
            for pair in raw['actions']:
                obs=sim.obs(s);history.append(obs);targets=list(carriers(obs['players'],1-s))
                if targets:
                    reference=prior.predict(list(history))
                    for uid,_,_,features in targets:
                        actual=pair[1-s].get(uid)
                        y=_DIRECTION.index(actual or None)
                        previous=np.array(reference[uid]['move'])
                        logits=features@w;v=np.exp(logits-logits.max());v/=v.sum()
                        sums += [-math.log(previous[y]),-math.log(v[y]),int(previous.argmax()==y),int(v.argmax()==y)]
                        n+=1
                sim.step(*pair)
            rows.append(dict(seed=raw['seed'],arm=path.parent.name,seat=s,n=n,sums=sums.tolist()))
    a=np.sum([r['sums'] for r in rows],0)/sum(r['n'] for r in rows)
    native.save(HERE/'reference.json',dict(rows=rows,prior_nll=a[0],head_nll=a[1],prior_accuracy=a[2],head_accuracy=a[3]))
    print(a)
if __name__=='__main__':main()
