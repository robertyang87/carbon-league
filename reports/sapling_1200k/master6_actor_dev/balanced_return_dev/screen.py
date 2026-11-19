from common import HERE, assert_cpu, finish, net, open_state, read, sha, write
import fcntl
import statistics


def main():
    assert_cpu()
    me=net('it20_self');foes={n:net(n) for n in ('it20_self','frozen_it50')}
    frozen={p:sha(HERE/p) for p in ('maps.json','screen_protocol.json','common.py','screen.py','worlds.py')}
    output=[]
    for seed in read(HERE/'maps.json')['screen']:
        for historically_focal in (0,1):
            for t in (225,275):
                seat=historically_focal if t==225 else 1-historically_focal
                w,ff,prefix,source=open_state(seed,historically_focal,t)
                for name,foe in foes.items():
                    state=dict(seed=seed,historical_seat=historically_focal,focal_seat=seat,t=t,opponent=name,source=source,registration=frozen)
                    p=HERE/'screen'/f'{seed}_{historically_focal}_{t}_{name}.json.gz'
                    if p.exists(): row=read(p);assert row['state']==state
                    else:
                        nets=[me,foe] if seat==0 else [foe,me]
                        row=dict(state=state,**finish(seed,prefix,w,ff,nets));write(p,row)
                    r=row['official'];output.append(dict(seed=seed,seat=seat,t=t,opponent=name,
                        margin=r['margin_by_seat'][seat],u=r['u_by_seat'][seat],receipt=sha(p)))
                print('screened',seed,seat,t,flush=True)
    scores={}
    for name in foes:
        a=[r for r in output if r['opponent']==name];m=[r['margin'] for r in a];u=[r['u'] for r in a]
        interior=sum(.05<=v<=.95 for v in u)/len(u)
        scores[name]=dict(eligible=min(m)<0<max(m) and interior>=.5 and len(set(m))>=2,non_saturated_fraction=interior,
              near_margin_fraction=sum(abs(v)<=1000 for v in m)/len(m),mean_absolute_margin=statistics.mean(map(abs,m)),
              focal_wdl=[sum(v>0 for v in m),sum(v==0 for v in m),sum(v<0 for v in m)])
    valid=[n for n in foes if scores[n]['eligible']]
    picked=max(valid,key=lambda n:(scores[n]['near_margin_fraction'],-scores[n]['mean_absolute_margin'],n=='frozen_it50')) if valid else None
    write(HERE/'screen.json',dict(rows=output,summary=scores,selected=picked,registration=frozen))
    print(picked,scores,flush=True)

if __name__=='__main__':
    with (HERE/'.screen.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);main()
