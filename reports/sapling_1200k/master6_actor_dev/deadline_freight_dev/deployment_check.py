"""Verify deployed decisions on the two historical freight-loss tapes before spending a new map."""
from rescue import HERE
import sys
sys.path.insert(0,str(HERE.parent/'tree_response_training'))
import select_roots as sr
from rescue import choose
def check():
    results=[]
    for c in sr.load(HERE.parent/'bank_under_pressure_dev/audit.json'):
        if not c['losses']:continue
        path=sr.ROOT/c['source'];raw=sr.load(path);seat=c['seat']
        sim=sr.FastCarbon(raw['game_configuration']).reset(seed=c['seed'])
        mod=sr.live.module(HERE/'candidate.py','freight_deployment_check')
        seen=[]
        for t,pair in enumerate(raw['actions']):
            ob=sim.obs(seat);orders=mod.agent(ob,raw['game_configuration'])
            if orders!=pair[seat]:
                seen.append(dict(t=t,changes={k:[pair[seat].get(k),orders.get(k)] for k in set(orders)|set(pair[seat]) if orders.get(k)!=pair[seat].get(k)}))
            sim.step(*pair)
        results.append(dict(seat=seat,source=c['source'],changes=seen))
    sr.retain(HERE/'deployed_actions.json',results)
    want=sr.load(HERE/'seat1/live_protocol.json')['root']
    assert any(w['t']==want['t'] and w['changes'].get(want['worker'])==[None,want['alternative']] for w in results[1]['changes'])
if __name__=='__main__':check()
