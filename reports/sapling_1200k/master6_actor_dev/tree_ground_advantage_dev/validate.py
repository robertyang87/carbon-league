"""Check the frozen place-or-move decision against archived paired genuine Master3s2 continuations."""
from fit import HERE,ACTOR,sr,CashEvents
from decision import update_memory,consider
from collections import defaultdict
from statistics import mean
import math

def simulate(case,row,raw,arm,snap):
    a=row['arms'][arm];s=row['seat'];sim=sr.FastCarbon(raw['game_configuration']);sim.set_state(snap);sim.cash=CashEvents(sim.cash)
    rec=sr.load(case/('live_'+a['native_arm']+'.json'));tape_path=case/rec['action_tape']
    assert sr.sha(tape_path)==a['archived_tape_sha256']==rec['action_sha256'] and rec['score']==a['official_physical_scores']
    tape=sr.load(tape_path);ground=tax=0.;team=[0.,0.]
    for j,joint in enumerate(tape):
        t=sim.step_count;sim.cash.events.clear();sim.step(*joint)
        for kind,who,val,*x in sim.cash.events:
            if kind=='tree_absorption':
                team[who]+=val-x[2]
                if (x[0]==a['tree_id'] and who==s and a['purchase_action_t'] is not None and
                    t>=a['purchase_action_t'] and (a['end_first_ownership_observation_t'] is None or t+1<=a['end_first_ownership_observation_t'])):
                    ground+=val-x[2];tax+=x[2]
    assert abs(ground+tax-a['absorption'])<.0001,(row['seed'],row['t'],arm,ground,tax,a['absorption'])
    return dict(gross_ground=ground,tax=tax,first_ground_net=ground-a['debit'],ground_team_margin=team[s]-team[1-s],
                official=a['official_margin'],survived=a['survived_root'],fee=a['debit'])

def run():
    rs=sr.load(ACTOR/'two_arm_plant_economy_dev/native_eval_rows.json.gz');w=sr.load(HERE/'weights.json');train=set(w['train_seeds'])
    native=[];cache={}
    for row in sorted(rs,key=lambda r:(r['seed'],r['seat'],r['t'])):
        assert row['seed'] not in train
        case=sr.ROOT/row['case'];q=sr.load(case/'live_protocol.json');src=q['record']['source'];s=row['seat'];t=row['t']
        if (src,s) not in cache:
            path=sr.ROOT/src;assert sr.sha(path)==row['source_sha256'];raw=sr.load(path)
            cache[src,s]=(raw,sr.FastCarbon(raw['game_configuration']).reset(seed=raw['seed']),{})
        raw,sim,mem=cache[src,s];cfg=raw['game_configuration']
        while sim.step_count<t:
            update_memory(sim.obs(s),mem);sim.step(*raw['actions'][sim.step_count])
        obs=sim.obs(s);update_memory(obs,mem);order=row['parent_pick'];reason=None
        cand=consider(obs,mem,row['own_root_joint']['move'],row['uid'],cfg,w['coef']) if order=='move' else None
        if cand and cand['accept']:order='stay'
        rr=dict(seed=row['seed'],case=row['case'],t=t,seat=s,worker=row['uid'],panel=row['historical_use'],parent=row['parent_pick'],
                pick=order,decision=cand,map=row['map'])
        if order!=row['parent_pick']:
            rr['arms']={arm:simulate(case,row,raw,arm,sim.get_state()) for arm in ('stay','move')}
        else:
            rr['same_terminal_margin']=row['arms'][order]['official_margin']
        native.append(rr)
    sr.retain(HERE/'paired_decisions.json.gz',native)
    sums=[]
    for split in ('all','train','hold','outside'):
        rows=[r for r in native if split=='all' or r['panel']==split];changed=[r for r in rows if r['pick']!=r['parent']];sums.append(dict(split=split,roots=len(rows),maps=len({r['map'] for r in rows}),priced=sum(r['decision'] is not None for r in rows),changed=len(changed),delta={k:sum(r['arms']['stay'][k]-r['arms']['move'][k] for r in changed)/len(rows) for k in ('first_ground_net','ground_team_margin','official')}))
    sr.retain(HERE/'paired_summary.json',sums);print(sums)
if __name__=='__main__':run()
