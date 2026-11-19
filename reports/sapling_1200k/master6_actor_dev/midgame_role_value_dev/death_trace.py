"""Trace the first recruited collector through actual payments or death, including the surrounding threats."""
from data import sr,HERE
from role import _D,_TO,_DIRECTION
def trace(seat):
    r=sr.load(HERE/f'audit_candidate_{seat}.json.gz')['details'];w=r['focal'];uid=w['uid']
    p=next((HERE/'games/candidate').glob(f'raw_*_{seat}.json.gz'));raw=sr.load(p)
    sim=sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']);trace=[]
    for t,cmd in enumerate(raw['actions']):
        obs=sim.obs(seat);own=obs['players'][seat];enemy=obs['players'][1-seat]
        entity=own[2].get(uid);record=None
        if w['t']<=t<min(w['t']+10,w['death'] or 299):
            record=dict(t=t,entity=entity,orders=cmd[seat].get(uid),enemies=[],safe_options=[])
            if entity:
                pos,cargo,_=entity;used={int(_TO[v[0],_DIRECTION.index(cmd[seat].get(k))]) for k,v in own[2].items() if k!=uid}
                hired={b for k,b in own[1].items() if cmd[seat].get(k)}
                foes=[(k,v) for k,v in enemy[2].items() if v[2]=='COLLECTOR' and _D[pos,v[0]]<=2]
                record['enemies']=[dict(uid=k,position=v[0],cargo=v[1],actual_order=cmd[1-seat].get(k)) for k,v in foes]
                for m,z in zip(_DIRECTION,_TO[pos]):
                    if z not in used|hired and z not in enemy[1].values() and all(_D[z,v[0]]>1 or v[1]>cargo for _,v in foes):
                        record['safe_options'].append(m or 'NONE')
        sim.step(*cmd)
        if record:
            record.update(survived=uid in sim.workers,collision=[v for v in sim.last_collisions if v[0]==uid]);trace.append(record)
    return dict(seat=seat,focal=uid,trace=trace)
if __name__=='__main__':
    records=[trace(s) for s in (0,1)];sr.retain(HERE/'new_worker_trace.json',records);print(records[0]['trace'][-1])
