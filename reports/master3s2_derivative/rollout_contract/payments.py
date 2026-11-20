"""Check natural historical receipts for the workers whose rollout decoder contract changed."""
from pathlib import Path
import gzip,json,zipfile
from probe import HERE,base,rr,fp
def main():
    panel=json.loads(gzip.decompress((HERE/'controlled.json.gz').read_bytes()))['public_actor_checks']
    results=[]
    for ar in json.loads((HERE/'protocol.json').read_bytes())['archives']:
        with zipfile.ZipFile(HERE.parent/'cold_start/development/hosted'/(ar+'.zip')) as z:
            reg=json.loads(z.read('cold_repair/registration.json'));turns=z.read('cold_repair/turns.jsonl').splitlines()
            frames=gzip.decompress(z.read('cold_repair/observations.jsonl.gz')).splitlines()
        seed=reg['seed'];side=reg['candidate_seat'];checks=[x for x in panel if x['seed']==seed and x['seat']==side and x['changed']]
        if not checks:continue
        sim=base.econ.FastCarbon(base.econ.config()).reset(seed=seed);sim.cash=base.econ.CashEvents(sim.cash)
        transactions=[]
        for t,(frame,turn) in enumerate(zip(frames,turns,strict=True)):
            f=json.loads(frame);q=json.loads(turn);obs=f['observations'][side]
            assert obs['players']==sim.obs(side)['players'] and base.digest(obs)==q['public_observation_sha256'][side]
            acts=[v['response']['action'] for v in q['calls']]
            sim.cash.events.clear();sim.step(*acts)
            transactions.extend(dict(t=t,event=e) for e in sim.cash.events if e[0]=='own_delivery')
        for case in checks:
            for d in case['unit_changes']:
                after=[r for r in transactions if r['t']>=case['t'] and r['event'][3]==d['uid']]
                results.append(dict(seed=seed,seat=side,t=case['t'],**d,first_real_deposit=after[0] if after else None))
    rr.save(HERE/'public_execution.json',dict(checks=results,meaning='Historical UID-scoped first actual deposit only, never a branch margin or marginal gain. A UID can repeat on multiple decision turns.'))
    print([(v['t'],v['uid'],v['first_real_deposit']) for v in results])
if __name__=='__main__':main()
