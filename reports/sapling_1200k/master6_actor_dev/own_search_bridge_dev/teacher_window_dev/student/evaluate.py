from pathlib import Path
from collections import Counter,defaultdict
import sys,json
import collect as C
from collect import HERE,P,B,ROOT,panel
def worker(arm):
    pack=B.PACK if arm=='parent' else HERE/'student.py';r=[]
    if arm=='student':assert B.read(HERE/'fit.json')['student']==B.sha(pack)
    for c in P['cards']:
        # All cards retained. Only the two fixed held terrains decide the gate.
        source=B.read(ROOT/c['source']);cfg=source['game_configuration'];m=B.live.module(pack,'native_student_pack');s=c['seat']
        sim=panel.FastCarbon(cfg).reset(seed=c['seed']);labels=B.read(HERE/f"card_{c['id']:02d}.json.gz")['records']
        for t,pair in enumerate(source['actions'][:158]):
            obs=sim.obs(s)
            if t==0 or t>=150:
                cmd=B.clean(m.agent(obs,cfg))
                if arm=='parent':assert cmd==B.clean(pair[s])
                if t>=150:
                    y=labels[t-150];assert y['t']==t
                    truth=y['teacher'];alive=set(obs['players'][s][1])|set(obs['players'][s][2]);assert set(truth)|set(cmd)<=alive
                    hit=Counter();total=Counter()
                    for uid in alive:
                        role=obs['players'][s][2][uid][2] if uid in obs['players'][s][2] else 'BASE'
                        hit[role]+=cmd.get(uid)==truth.get(uid);total[role]+=1
                    r.append(dict(case=c['id'],seed=c['seed'],seat=s,t=t,split=c['split'],public_digest=y['public_hash'],teacher=truth,actual=cmd,
                              matches=sum(hit.values()),slots=len(alive),matches_by_role=dict(hit),slots_by_role=dict(total),exact=truth==cmd))
            else:
                # In the frozen wrapper only Feat is carried between calls. Use the wrapper every evaluated step.
                m._FEAT.observe(obs)
            sim.step(*pair)
    B.live.save(HERE/(arm+'_entry.json.gz'),dict(pack=B.sha(pack),program=B.sha(Path(__file__)),rows=r))
def assess():
    a,b=(B.read(HERE/(arm+'_entry.json.gz'))['rows'] for arm in ('parent','student'))
    tables=defaultdict(Counter)
    for x,y in zip(a,b,strict=True):
        assert all(x[k]==y[k] for k in ('case','t','teacher','slots','public_digest'))
        for name in (x['split'],f"{x['split']}_seat{x['seat']}",str(x['seed'])):
            q=tables[name];q['roots']+=1;q['slots']+=x['slots'];q['changed']+=(x['actual']!=y['actual'])
            for label,z in [('parent',x),('student',y)]:
                q[label+'_correct']+=z['matches'];q[label+'_teams']+=z['exact']
                for role,ct in z['matches_by_role'].items():q[label+'_'+role]+=ct
    grids={str(c['seed']) for c in P['cards'] if c['split']=='held'}
    tests={k:tables[k]['student_correct']>tables[k]['parent_correct'] for k in sorted(grids|{'held_seat0','held_seat1'})}
    tests['teams']=tables['held']['student_teams']>=tables['held']['parent_teams']
    tests['both_seats_changed']=all(tables[f'held_seat{s}']['changed']>0 for s in (0,1))
    B.live.save(HERE/'results.json',dict(protocol=B.sha(HERE/'protocol.json'),fit=B.sha(HERE/'fit.json'),program=B.sha(Path(__file__)),
        entry={arm:B.sha(HERE/(arm+'_entry.json.gz')) for arm in ('parent','student')},by_group={k:dict(v) for k,v in tables.items()},tests=tests,gate_pass=all(tests.values())))
if __name__=='__main__':
    if len(sys.argv)>1:worker(sys.argv[1])
    else:assess()
