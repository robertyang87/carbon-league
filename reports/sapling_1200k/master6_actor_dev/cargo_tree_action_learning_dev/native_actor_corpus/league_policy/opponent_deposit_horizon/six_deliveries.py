"""Revisit six already-published outcomes at exact causal horizons, on four explicitly exposed games only."""
from collections import deque,defaultdict
import json
from pathlib import Path
import zipfile

import numpy as np
from predict import PublicDeposit,HERE,short,read,module
from native import sha,save,require,FastCarbon,CashEvents
LANE=HERE.parent/'natural_original';PAST=LANE/'fresh_repeat'
CASH=module('native_inspect_cash',LANE/'economy.py')


def run():
    old=read(PAST/'interception/summary.json')
    evidence=read(PAST/'hosted/settlement.json');model=PublicDeposit()
    require(evidence['workflow_run']==44625133102 and evidence['records_reconciled'],'wrong historic native run')
    cases=old['cases'];out=[]
    for g in evidence['games']:
        if g['arm']!='sampled_child':continue
        assert g['status']=='complete'
        seed,seat=g['seed'],g['seat'];tapes=next(q for q in evidence['archives'] if q['seed']==seed and q['seat']==seat)
        path=PAST/'hosted'/f"{tapes['id']}.zip"
        require(sha(path.read_bytes())==tapes['sha256'],'wrong retained host archive')
        with zipfile.ZipFile(path) as z:
            rec=z.read('sampled_child/receipt.json')
            tape=z.read('sampled_child/turns.jsonl')
        require(sha(rec)==g['receipt_sha256'] and sha(tape)==json.loads(rec)['turns_sha256'],'tape identity differs')
        sim=FastCarbon(CASH.config()).reset(seed=seed)
        require(sha(bytes(map(int,sim.carbon)))==g['terrain'],'map differs')
        sim.cash=CashEvents(sim.cash);hist=deque(maxlen=11);frozen=[];paid={}
        relevant=[q for q in cases if q['seed']==seed and q['seat']==seat]
        roots=defaultdict(list)
        for case in relevant:
            for h in (3,5,10):
                for mode,root in [('previous_geometry',case['pivot']-h),('actual_horizon',case['pivot']+1-h)]:
                    roots[root].append((h,mode,case))
        for step,line in enumerate(tape.splitlines()):
            row=json.loads(line);require(step==row['step']==sim.step_count,'step mismatch')
            obs=sim.obs(seat);hist.append(obs)
            if step in roots:
                rows=sorted(short.public_rows(list(hist)),key=lambda r:str(r[0]))
                predictions=model.arrays(np.stack([q[1] for q in rows])) if rows else None
                for h,mode,case in roots[step]:
                    k=(3,5,10).index(h);uid=case['target']
                    ranked=sorted(range(len(rows)),key=lambda j:(-rows[j][1][0]*predictions[0][j,k],str(rows[j][0])))
                    ids=[rows[j][0] for j in ranked]
                    detail=dict(seed=seed,seat=seat,worker=uid,t=step,h=h,pivot_action=case['pivot'],
                                receipt_at_state=case['pivot']+1,selection=mode,loaded_workers=len(rows),delivered_on_pivot=case['delivery'])
                    if uid in ids:
                        rank=ids.index(uid)+1;i=ranked[rank-1];x=rows[i][1]
                        detail.update(cargo=float(x[0]),home_distance=int(x[1]),p=float(predictions[0][i,k]),
                                      cargo_distance=float(predictions[1][i,k]),rank_amount=rank,
                                      top_uid=str(ids[0]),top_p=float(predictions[0][ranked[0],k]))
                    frozen.append(detail)
            sim.cash.events.clear();sim.step(*[c['response']['action'] for c in row['calls']])
            for ev in sim.cash.events:
                if ev[0]=='own_delivery' and ev[1]==1-seat and ev[2]>0:paid[step+1,ev[3]]=ev[2]
        for item in frozen:
            t,h,uid=item['t'],item['h'],item['worker']
            require(abs(paid[item['receipt_at_state'],uid]-item['delivered_on_pivot'])<1e-5,'different published actual payment')
            item['true_positive_payment_within_horizon']=sum(paid.get((j,uid),0) for j in range(t+1,t+h+1))
            out.append(item)
    require(len(out)==36,'published six-worker panel incomplete')
    return dict(observations=out,selected_after_outcome=True,training_rows=0,
                input_sha256=sha((PAST/'interception/summary.json').read_bytes()),
                note="The geometric study looked back from the action index. At that exact root the named deposit occurs one step outside the nominal H. The actual_horizon offsets from paid state and evaluates H settled joint transitions.")


if __name__=='__main__':save(HERE/'six_deliveries.json',run())
