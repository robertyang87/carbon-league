"""Settle all registered t298 roots and the first two changed t293 maps using the official referee."""
import copy
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

import panel

HERE=panel.HERE
FAMILY=panel.FAMILY
CHECK=FAMILY/'endgame_cash_projection/response_check'
sys.path.insert(0,str(CHECK))
import recovery

game=recovery.freight
live=recovery.live


def verified_equal(env, sim):
    board=env.states[0].observation
    assert board.players==sim.obs(0)['players'] and board.trees==sim.obs(0)['trees'] and board.full_carbon==sim.carbon
    assert [p[0] for p in board.players]==list(sim.cash)


def step(env,sim,actions):
    sim.cash.events.clear()
    sim.step(*actions)
    env.step(actions,[])
    verified_equal(env,sim)


def at_root(card,t,mods=None):
    reg,turns,frames=panel.sources(card)
    with zipfile.ZipFile(FAMILY/card['source']) as z:
        raw=z.read('parent/registration.json')
        receipt=json.loads(z.read('parent/receipt.json'))
    assert all(receipt['initialization'][k]['response']['sha256']==reg['files'][r]['sha256']
               for k,r in ((card['seat'],'candidate'),(1-card['seat'],'opponent')))
    assert receipt['initialization'][1-card['seat']]['response']['original']==dict(_TEMP=1.,_SEED=None,_S_ABORT=.58,_SEARCH_OK=True)
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/'registration.json';f.write_bytes(raw)
        live.configure(f,Path(td)/'unused',Path(td)/'unused.lock');env,cfg=live.official()
    sim=game.econ.FastCarbon(cfg).reset(seed=card['seed'])
    sim.cash=game.econ.CashEvents(sim.cash)
    bank=[5.,5.]
    trackers=[m.BeliefTracker(m._Bel,k) for k,m in enumerate(mods)] if mods else []
    prev=None
    for i,(frame,turn) in enumerate(zip(frames,turns,strict=True)):
        obs=[game.public(env,k,bank) for k in (0,1)]
        assert obs==frame['observations'] and [game.digest(o) for o in obs]==frame['public_observation_sha256']==turn['public_observation_sha256']
        assert turn['step']==frame['step']==i
        assert all(c['status']=='ok' for c in turn['calls'])
        orders=[c['response']['action'] for c in turn['calls']]
        if i==t:
            break
        for k,tr in enumerate(trackers):
            tr.step(obs[k]['players'],i,obs[k]['carbon'])
            tr.set_actions(obs[k]['players'],orders[k])
        prev=obs[0]['players']
        step(env,sim,orders)
        bank=turn['overage_remaining_seconds']
    assert not env.done
    env.steps[:-1]=[None]*t;env.logs=[]
    return env,sim,cfg,obs,bank,orders,trackers,prev,receipt,turn


def terminal(env,s):
    assert env.done and all(st.status=='DONE' for st in env.states)
    scores=[st.reward for st in env.states]
    return dict(scores=scores,margin=round(scores[s]-scores[1-s],2),
                win=scores[s]>scores[1-s],draw=scores[s]==scores[1-s])


def signed_panel(card):
    path=HERE/'roots'/f"{card['seed']}_{card['seat']}.json.gz"
    raw=path.read_bytes();p=json.loads(gzip.decompress(raw))
    assert p['protocol']==panel.LOCK and all(p[k]==card[k] for k in card)
    assert [r['step'] for r in p['rows']]==panel.PLAN['steps']
    return p['rows'],panel.sha(raw)


def last(card,row,sha):
    s=card['seat'];env,sim,cfg,obs,bank,orders,_,_,receipt,_=at_root(card,298)
    assert game.digest(obs[s])==row['public_sha'] and orders[1-s]==row['real_examiner_root']
    frozen=game.official_state.take(env,bank)
    arms={}
    for name,order in [('historical',orders[s]),('M1',row['M1']['issued']),('M2',row['M2']['issued'])]:
        game.official_state.restore(env,frozen);shadow=copy.deepcopy(sim)
        pair=orders[:];pair[s]=order
        step(env,shadow,pair)
        arms[name]=dict(orders=pair,**terminal(env,s))
    assert arms['historical']['scores']==receipt['official_scores_by_seat']
    return dict(seed=card['seed'],seat=s,step=298,public_sha=row['public_sha'],source_sha=card['sha256'],
                panel_sha=sha,different_issued=arms['M1']['orders']!=arms['M2']['orders'],arms=arms,
                M2_minus_M1_margin=round(arms['M2']['margin']-arms['M1']['margin'],2))


def suffix(card,row,panelsha,ordinal,*,variants=('M1','M2'),scope=HERE,protocol=panel.LOCK):
    s=card['seat'];t=row['step'];assert t in (282,293)
    paths=[FAMILY.parents[1]/'data/master3s2_source.py']*2;paths[s]=panel.PARENT
    assert [panel.sha(p.read_bytes()) for p in paths]==[panel.PLAN['parent' if k==s else 'examiner'] for k in (0,1)]
    mods=[game.load('late_fixed_side'+str(k),p) for k,p in enumerate(paths)]
    assert all(m._SEED is None and m._TEMP==1. and m._S_ABORT==.58 and m._S_M==m._S_MMAX==1 and m._SEARCH_OK for m in mods)
    for m in mods:m._net()
    env,sim,cfg,obs,bank,archive,trackers,previous,_,history=at_root(card,t,mods)
    assert game.digest(obs[s])==row['public_sha'] and archive[1-s]==row['real_examiner_root']
    seeds=[row['policy_seed'] if k==s else row['policy_seed']^0x7d4a1f6b for k in (0,1)]
    # Historic private draws and cost caches are unavailable. Establish one common
    # post-root prefix per side before splitting. Only witnessed enemy commands enter the referee.
    init=[]
    for k,m in enumerate(mods):
        m._ST=dict(tr=trackers[k],prev=previous,rng=game.np.random.RandomState(seeds[k]))
        m._S_COST[:]=row['cost'] if k==s else [.001,.002,.001,.002]
        a,w=game.measured(m,obs[k],cfg)
        m._ST['tr'].set_actions(obs[k]['players'],archive[k])
        init.append(dict(local_reconstructed_command=a,clock=w))
    state=[copy.deepcopy(game.original_state.dynamic(m)) for m in mods]
    configs=[copy.deepcopy(m._RX_CFG[1]) for m in mods]
    root=game.official_state.take(env,bank)
    out={}
    first,second=variants
    order=list(variants if ordinal==0 else reversed(variants))
    for label in order:
        remaining=game.official_state.restore(env,root);shadow=copy.deepcopy(sim)
        for k,m in enumerate(mods):
            for var,val in copy.deepcopy(state[k]).items():setattr(m,var,val)
            m._RX_CFG[:]=[cfg,copy.deepcopy(configs[k])]
        action=row[label]['issued'];mods[s]._ST['tr'].set_actions(obs[s]['players'],action)
        tape=[]
        while not env.done:
            views=[game.public(env,k,remaining) for k in (0,1)]
            i=views[0]['step'];digests=[game.digest(v) for v in views]
            if i==t:
                moves=archive[:];moves[s]=action
                r=history['calls'][1-s]['response']
                clocks=[None,None];clocks[s]=row[label]['clock'];clocks[1-s]=dict(wall=r['public_wall_seconds'],cpu=r['public_cpu_seconds'],search=r.get('search_last'))
            else:
                replies=[game.measured(m,views[k],cfg) for k,m in enumerate(mods)]
                moves=[v[0] for v in replies];clocks=[v[1] for v in replies]
            remaining=[b-max(0.,v['wall']-1.) for b,v in zip(remaining,clocks)]
            step(env,shadow,moves)
            tape.append(dict(step=i,public_sha=digests,orders=moves,clocks=clocks,remaining_bank=remaining,end_cash=list(shadow.cash)))
        result=terminal(env,s)
        path=scope/'branches'/f"{card['seed']}_{s}_{label}.json.gz"
        evidence=dict(ticket=card,panel_sha=panelsha,protocol=protocol,root_order=label,
                      root_order_source='completed native panel issuance and original archived simultaneous opposing order',
                      continuation_by_seat=[panel.sha(p.read_bytes()) for p in paths],common_rng_by_seat=seeds,
                      common_root_warmup=init,host_cpu=sorted(os.sched_getaffinity(0)),tape=tape,**result)
        panel.put(path,evidence)
        out[label]=dict(transitions=len(tape),evidence=path.relative_to(scope).as_posix(),sha256=panel.sha(path.read_bytes()),**result)
    return {'seed':card['seed'],'seat':s,'step':t,'branch_order':order,'arms':out,
            f'{second}_minus_{first}_margin':round(out[second]['margin']-out[first]['margin'],2)}


def main():
    assert len(os.sched_getaffinity(0))==1
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    os.environ['CARBON_DEBUG']='1'
    cards=sorted(panel.PLAN['tickets'],key=lambda c:(c['seed'],c['seat']))
    assert len(cards)==8 and len({(c['seed'],c['seat']) for c in cards})==8
    end=[];choices=[];elig=[]
    for card in cards:
        rows,sha=signed_panel(card);r=next(q for q in rows if q['step']==293)
        end.append(last(card,next(q for q in rows if q['step']==298),sha))
        changed=r['M1']['issued']!=r['M2']['issued']
        elig.append(dict(seed=card['seed'],seat=card['seat'],changed=changed))
        if changed and all(c['seed']!=card['seed'] for c,_,_ in choices) and len(choices)<2:choices.append((card,r,sha))
    (HERE/'branches').mkdir(exist_ok=True)
    suffixes=[suffix(c,r,h,i) for i,(c,r,h) in enumerate(choices)]
    panel.put(HERE/'terminal.json',dict(protocol_sha=panel.LOCK,physical_referees='official plus FastCarbon cash, workers, trees and full carbon after every transition',
              root_source='actual original opponent simultaneous response for both arms',t298=end,t293_eligibility=elig,t293_responses=suffixes,
              note='Local one-core wall and CPU are diagnostic. Natural temperature-one continuations share reconstructed starting RNG, not the historical private RNG. Complete counterfactual terminal margins from old roots are not whole-game wins.'))
    print([(r['seed'],r['seat'],r['M2_minus_M1_margin']) for r in end],suffixes,flush=True)


if __name__=='__main__':main()
