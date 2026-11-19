from scan import HERE,sr
from contact import choose
from occupy import harvest
import sys
sys.path.insert(0,str(HERE.parent/'bank_under_pressure_dev'))
from turning_point_ledger import CashEvents
import timing,periods
def check():
    rows=[]
    for r in sr.load(HERE/'scores.json'):
        s=r['seat'];src=HERE/'games'/r['arm']/f"raw_{r['seed']}_{s}.json.gz";raw=sr.load(src)
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=r['seed']);sim.cash=CashEvents(sim.cash)
        parent=sr.live.module(HERE.parent/'planter_capture_full_dev/candidate.py','patch_parent')
        prev=root=None;changes=[];deposits=[];early=[]
        for t,cmd in enumerate(raw['actions']):
            obs=sim.obs(s);orders=parent.agent(obs,raw['game_configuration']);expected=dict(orders)
            if r['arm']=='candidate':
                if root is None and 120<=t<=200:
                    root=choose(obs,orders,prev)
                    if root:
                        root['site']=obs['players'][1-s][2][root['target']][0];expected[root['worker']]=root['alternative']
                elif root and t==root['t']+1:expected=harvest(obs,orders,root)
            assert expected==cmd[s],(s,t)
            if expected!=orders:changes.append(t)
            prev=obs;sim.cash.events.clear();sim.step(*cmd)
            if root and t>=root['t']:
                ids=(root['worker'],root['target'])
                deposits.extend([t,*e] for e in sim.cash.events if e[0]=='own_delivery' and e[3] in ids)
                if t<=root['t']+4:
                    early.append(dict(t=t,workers={k:[sim.workers[k].pos,sim.workers[k].carbon] if k in sim.workers else None for k in ids}))
        assert any(raw['extinct_by_seat']) or list(sim.cash)==r['score_by_seat']
        rows.append(dict(**r,edits=changes,root=root,first_five=early,deposits=deposits,periods=periods.compare(src,s)['windows']))
    timing.HERE=HERE;timing.check()
    sr.retain(HERE/'full_ledger.json',rows)
if __name__=='__main__':check()
