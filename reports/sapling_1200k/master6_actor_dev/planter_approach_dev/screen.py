"""Screen both physical seats on frozen native tapes, then replay each first launch against recorded rival orders."""
from collections import Counter
import copy
import gzip
import hashlib
import json
import sys
from build import BASE,HERE,ROOT,PARENT,rebuild,sha
sys.path.insert(0,str(BASE/'root_recovery_branches'))
import live
from branches import clean
sys.path.insert(0,str(BASE))
from turning_point_ledger import FastCarbon,CashEvents

T=HERE.parent/'tree_response_training'


def read(path):
    data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def retain(path,value):
    if path.exists():
        if read(path)!=value:raise ValueError(f'changed receipt {path}')
    else:live.save(path,value)


def no_cash_or_enemy(obj,q,joint):
    cases={}
    for name in ('no_cash','threat_at_destination','stale_empty_cell'):
        alt=copy.deepcopy(obj)
        uid,cell=q['uid'],q['cell']
        if name=='no_cash':
            alt.market.cash=0
        elif name=='threat_at_destination':
            alt.enemy=copy.deepcopy(alt.enemy)
            worker=next(iter(alt.enemy[2]),None)
            if worker is None:continue
            alt.enemy[2][worker][0]=cell
        else:
            if cell in alt.market.opp_trees:continue
            alt.market.last[:]=-1000
        alt.values={}
        cases[name]=alt.quote(cell,uid,joint,q['distance']) is None
    if not all(cases.values()):raise ValueError('unsafe historical counterexample')
    return cases


def scan(game):
    raw=read(ROOT/game['raw_path'])
    cfg=raw['game_configuration'];s=game['seat'];seed=game['seed']
    model=live.module(HERE/'agent.py','public_approach')
    sim=FastCarbon(cfg).reset(seed=seed)
    rows=[];first=None;turns=0;moving=0;reasons=Counter()
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(s);recorded=clean(pair[s])
        if t==0:
            if clean(model.agent(obs,cfg))!=recorded:raise ValueError('wrong v1 init')
        else:
            state=model._APP
            state.observe(obs,cfg)
            if 50<=t<=180:
                turns+=1
                moving+=sum(w[2]=='PLANTER' and uid in recorded for uid,w in state.own[2].items())
                q=state.proposal(recorded)
            else:q=None
            if q:
                out=clean(model._approach_underlying(obs,cfg));event=state.last
                reasons[event['reason']]+=1
                changed=[u for u in set(out)|set(recorded) if out.get(u)!=recorded.get(u)]
                if changed!=([q['uid']] if event.get('changed') else []):
                    raise ValueError('intervention violates original root or contribution contract')
                aud={'t':t,**q,'reason':event['reason'],'issued':event.get('issued'),
                     'old':recorded[q['uid']],'changed':bool(changed)}
                rows.append(aud)
                if changed and first is None:
                    first={**aud,'parent_commands':recorded,'candidate_commands':out,
                           'counterexamples':no_cash_or_enemy(state,q,recorded)}
                # Subsequent roots get individually screened along the original, never an unexecuted memory.
                state.ticket=None;state.used=False
            else:model._FEAT.observe(obs)
        sim.step(*pair)
    if turns!=131 or sim.step_count!=len(raw['actions']):raise ValueError('incomplete historical tape')
    return dict(seed=seed,seat=s,steps=sim.step_count,window=turns,moving_planter_decisions=moving,
                raw_sha256=game['raw_sha256'],board_sha256=game['map_sha256'],reasons=dict(reasons),offers=rows,first=first)


def short(game,root,arm):
    raw=read(ROOT/game['raw_path']);cfg=raw['game_configuration'];s=game['seat'];t0=root['t']
    model=live.module((HERE/'agent.py') if arm=='approach' else PARENT,'bound_short_'+arm)
    sim=FastCarbon(cfg).reset(seed=game['seed']);sim.cash=CashEvents(sim.cash)
    no_student=live.module(PARENT,'single_expert_ablation');no_student._HYBRID.update(expert_weight=1.,student_weight=0.)
    contribution=0;rows=[];uid=root['uid'];cell=root['cell'];paid=0.
    for t,pair in enumerate(raw['actions'][:t0+5]):
        obs=sim.obs(s)
        if t<t0:
            if t==0:
                if clean(model.agent(obs,cfg))!=clean(pair[s]):raise ValueError('initial mismatch')
            else:
                model._FEAT.observe(obs)
                if arm=='approach':model._APP.observe(obs,cfg)
            if clean(no_student.agent(obs,cfg))!=clean(pair[s]):contribution+=1
            sim.step(*pair);continue
        commands=clean(model.agent(obs,cfg))
        if t==t0 and commands!=(root['candidate_commands'] if arm=='approach' else root['parent_commands']):
            raise ValueError('first full branch command differs')
        joint=[dict(a) for a in pair];joint[s]=commands
        unknown=sorted(set(joint[1-s])-set(obs['players'][1-s][2])-set(obs['players'][1-s][1]))
        absent_orders=sorted(set(obs['players'][1-s][2])-set(pair[1-s]))
        sim.cash.events.clear();sim.step(*joint)
        items=[list(e) for e in sim.cash.events if e[0] in ('plant','seize') and e[1]==s and e[3]==uid and e[4]==cell]
        paid+=sum(-e[2] for e in items)
        worker=sim.obs(s)['players'][s][2].get(uid)
        rows.append({'t':t,'command':commands.get(uid,'NONE'),'after_cell':worker[0] if worker else None,
                     'settled_payments':items,'rival_tape_unavailable_or_dead_orders':unknown,
                     'rival_without_explicit_order':absent_orders,
                     'decision':model._APP.last if arm=='approach' else None})
    return dict(steps=rows,earned_target_arrival=any(r['after_cell']==cell for r in rows),cash_paid_at_target=paid,
                prefix_roots_with_actual_sapling_action_contribution=contribution)


def main():
    plan=read(HERE/'protocol.json');manifest=read(HERE/'manifest.json')
    if rebuild()[1]!=manifest or sha(HERE/'agent.py')!=manifest['pack_sha256'] or sha(T/'source_receipt.json')!=plan['source_receipt']:
        raise ValueError('wrong candidate or training corpus')
    certified=read(T/'source/official_replay.json')['runs']['source']['games']
    rows=[]
    for game in plan['public_source_games']:
        match,=[x for x in certified if (x['seed'],x['seat'])==(game['seed'],game['seat'])]
        if sha(ROOT/game['raw_path'])!=match['transcript_sha256'] or match['map_sha256']!=game['map_sha256'] or match['transcript_sha256']!=game['raw_sha256']:
            raise ValueError('uncertified source')
        row=scan(game);first=row['first']
        if first:
            row['fixed_opponent_tape']={arm:short(game,first,arm) for arm in ('parent','approach')}
        rows.append(row);print(game['seed'],game['seat'],len(row['offers']),row['reasons'],first and first['t'],flush=True)
    active=[r for r in rows if r['first']]
    eligible={r['seat'] for r in active}=={0,1} and any(r['fixed_opponent_tape']['approach']['cash_paid_at_target']>0 for r in active)
    native=next((r for key in plan['native_order'] for r in active if [r['seed'],r['seat']]==key),None) if eligible else None
    retain(HERE/'screen.json.gz',dict(protocol_sha256=sha(HERE/'protocol.json'),agent_sha256=sha(HERE/'agent.py'),script_sha256=sha(HERE/'screen.py'),games=rows,
        eligible_for_native=eligible,native_root=[native['seed'],native['seat'],native['first']['t']] if native else None))
    print('original eligible',eligible,flush=True)


if __name__=='__main__':main()
