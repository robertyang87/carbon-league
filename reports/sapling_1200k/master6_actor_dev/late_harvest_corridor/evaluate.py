"""Three paired continuations per public rival model, with chronological root selection and actual cash attribution."""
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import data
import policy

HERE=Path(__file__).resolve().parent
ROOT=data.ROOT
PROTO='1d6069ed2af8a3afb7df2c3d826ff50fe43031b02519ac32ef9a3ca0b4139c1f'
REF=data.reference()
PACK=ROOT/'data/weights/sapling_1200k/terminal_return_fit_v1.py'
PROXY=HERE.parent/'endgame_queue_dev/policy.py'
FOCAL=HERE.parent/'late_workforce_dev/focal_life.py'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


rival=module('corridor_fixed_public_opponent',PROXY)
worker=module('corridor_audited_collisions',FOCAL)
require,sha,read,write=REF.require,REF.sha,REF.read,REF.write


def clean(orders):
    return {u:c for u,c in orders.items() if c and c!='NONE'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def inputs():
    require(sha(HERE/'protocol.json')==PROTO,'protocol changed')
    receipt=read(HERE/'data_receipt.json'); registry=read(HERE/'source_registry.json')
    require(registry['protocol_sha256']==PROTO and receipt['source_registry_sha256']==sha(HERE/'source_registry.json') and
            receipt['public_sha256']==sha(HERE/'data.json.gz') and receipt['audit_sha256']==sha(HERE/'audit.json.gz') and
            receipt['source_sha256']==sha(HERE/'data.py') and registry['pack_sha256']==sha(PACK),'data or actor binding')
    files=[Path(__file__),HERE/'data_receipt.json',HERE/'policy.py',PROXY,FOCAL,data.STUDY,ROOT/'code/fastsim.py']
    return {'protocol_sha256':PROTO,'parent_sha256':sha(PACK),'files':{str(f.relative_to(ROOT)):sha(f) for f in files}},registry['games']


def select():
    require(sha(HERE/'protocol.json')==PROTO,'protocol changed')
    rows=[];chosen={}
    for split in ('dev','eval'):
        for key,obs,past,own,cfg in data.public_roots(split):
            seed,seat,t=key; h=cfg['episodeSteps']-1-t
            if not 5<=h<=12:continue
            selected,plan,meta=policy.choose(obs,past,own,cfg)
            diffs={u:[own.get(u),selected.get(u)] for u in sorted(own.keys()|selected.keys()) if clean(own).get(u)!=clean(selected).get(u)}
            require(bool(diffs)==(meta['status']=='start') and (not diffs or set(diffs)=={meta['target_worker']}),'start/action mismatch')
            row={'split':split,'seed':seed,'seat':seat,'t':t,'remaining':h,'changes':diffs,'meta':meta,'plan':plan,
                 'public_sha256':digest(obs),'baseline':own,'candidate':selected}
            rows.append(row)
            if diffs and ((seed,seat) not in chosen or chosen[seed,seat]['t']>t):chosen[seed,seat]=row
    require(len(rows)==128 and len({(r['seed'],r['seat'],r['t']) for r in rows})==128,'public launch coverage')
    return rows,chosen


def gross_harvest(sim,before,after,uid,joint):
    w=sim.workers.get(uid)
    if not w or not w.is_collector or joint[w.player].get(uid) in sim._move_nbr or w.pos in sim.base_at:return 0.
    trees=[age for pl in before['players'] for pos,age in pl[3].values() if pos==w.pos]
    if trees and trees[0]<sim.config['treeLifespan']:return 0.
    stock=sim.config['co2FrmWithered'] if trees else before['carbon'][w.pos]
    collectors=sum(row[2]=='COLLECTOR' for row in after['players'][w.player][2].values())
    return stock*max(0.,sim.config['initialCollectRate']-collectors*sim.config['collectDecreaseRate'])


class TaggedCargo:
    """Follow the focal starting cargo and its subsequently mined additions across either side's carriers."""
    def __init__(self,uid,seat,opening):
        self.uid=uid;self.seat=seat;self.issued=Counter(principal=opening,fresh=0.)
        self.held={uid:Counter(principal=opening)};self.closed={'fresh':Counter(),'principal':Counter()}

    def step(self,sim,before,counters,joint,account):
        after=sim.full_observation(); units=worker.positions(sim,before,counters,joint,account['events'])
        crashes=worker.collisions(sim,units); lost={e['victim']:e for e in crashes}
        gained=Counter()
        for e in crashes:
            if e['winner']:gained[e['winner']]+=e['cargo']
        by_uid={e[3]:e for e in account['events'] if e[0] in ('tree_collision_cash','base_collision_cash','own_delivery','opponent_worker_delivery_to_this_base')}
        surviving=defaultdict(Counter); landed=Counter()
        def book(kind,name,amount):
            if amount:self.closed[kind][name]+=amount;landed[kind+'/'+name]+=amount
        for uid,tags in self.held.items():
            p,cell,_,_=units[uid];moved=joint[p].get(uid) in sim._move_nbr
            fraction=1.-sim.config['moveCost'] if moved else 1.
            for tag,amt in tags.items():
                book(tag,'move_fee',amt*(1.-fraction)); mass=amt*fraction
                if uid not in lost:surviving[uid][tag]+=mass;continue
                e=lost[uid]
                if e['winner']:surviving[e['winner']][tag]+=mass
                elif e['recipient_seat'] is None:book(tag,'ground_drop_or_overflow',mass)
                elif mass:
                    receipt=by_uid[uid];prefix='own' if receipt[1]==self.seat else 'rival'
                    book(tag,prefix+('_bank' if receipt[0]=='base_collision_cash' else '_tree_collision'),mass)
        fresh=gross_harvest(sim,before,after,self.uid,joint)
        if fresh:surviving[self.uid]['fresh']+=fresh;self.issued['fresh']+=fresh
        old_trees={tid for pl in before['players'] for tid,(_,age) in pl[3].items() if age<sim.config['treeLifespan']}
        next_held={}
        for uid,tags in surviving.items():
            p,cell,load,_=units[uid];total=load+gained[uid]
            if cell in sim.base_at and total>0:
                owner=sim.base_at[cell][0];fee=sim.config['smeltCost']
                if owner!=p and total<=fee:
                    for tag,v in tags.items():book(tag,'hostile_base_removal',v)
                    continue
                if total>=fee:
                    credit=total-fee
                    if credit:require(uid in by_uid and math.isclose(by_uid[uid][2],credit,abs_tol=1e-7),'bank attribution')
                    for tag,v in tags.items():
                        book(tag,'smelt_fee',v*fee/total)
                        book(tag,'own_bank' if owner==self.seat else 'rival_bank',v*credit/total)
                    continue
            tax=sim.config['collectorAbsorptionRate']*sum(tid in old_trees and (cell==pos or cell in sim._nbr8[pos])
                  for tid,(pos,_) in after['players'][1-p][3].items())
            require(0<=tax<=1,'total tax out of range')
            residue=Counter()
            for tag,v in tags.items():
                book(tag,'rival_tree_tax' if p==self.seat else 'own_tree_tax',v*tax)
                if v*(1-tax)>0:residue[tag]=v*(1-tax)
            if residue:
                require(uid in sim.workers and sum(residue.values())<=sim.workers[uid].carbon+1e-6,'attributed cargo above real inventory')
                next_held[uid]=residue
        self.held=next_held
        for tag,total in self.issued.items():
            require(math.isclose(total,sum(self.closed[tag].values())+sum(r[tag] for r in self.held.values()),abs_tol=1e-6), 'principal/fresh stock-flow closure')
        return {'gross_focal_harvest':fresh,'posted_this_step':dict(landed),'focal_collisions':[e for e in crashes if e['winner']==self.uid or e['victim']==self.uid]}

    def summary(self):
        return {tag:{'issued':v,'fates':dict(self.closed[tag]),'inventory_at_end':sum(r[tag] for r in self.held.values()),
                'own_bank':self.closed[tag]['own_bank']} for tag,v in self.issued.items()}


def warm_start(agent,game,root,truth):
    seed,seat,t=game['seed'],game['seat'],root['t'];path=ROOT/game['path']
    require(sha(path)==game['raw_sha256'],'source transcript digest')
    raw=read(path);cfg=truth['config'];sim=REF.FastCarbon(cfg).reset(full_carbon=raw['initial_full_carbon'])
    require(REF.digest(bytes(int(v) for v in sim.carbon))==game['terrain_sha256'],'prior map checksum')
    past=[{},{}]
    for i in range(t):
        joint=raw['actions'][i]
        for p in (0,1):
            public=sim.obs(p)
            if p==seat:
                if i==0:require(clean(agent.agent(public,cfg))==clean(joint[seat]),'v1 opening mismatch')
                else:agent._FEAT.observe(public)
            if i>=t-12:past[p][i]=public
        sim.step(*joint)
    restored=REF.FastCarbon(cfg).set_state(truth['state'])
    require(sim.get_state()==restored.get_state() and list(truth['joint'])==raw['actions'][t]
            and digest(sim.obs(seat))==root['public_sha256'], 'true root reconstruction')
    return deepcopy(agent._FEAT),deepcopy(agent._COUNTS),past


def benefit(scores,seat):
    margin=scores[seat]-scores[1-seat]
    w=1. if margin>0 else 0. if margin<0 else .5
    return .5*w+.5/(1+math.exp(-max(-300.,min(300.,margin/1000.))))


def track_arm(agent,origin,truth,root,role,variant,cache):
    seat=root['seat'];enemy=1-seat;cfg={k:v for k,v in truth['config'].items() if k!='randomSeed'}
    uid=root['meta']['target_worker'];t=root['t'];sim=REF.FastCarbon(cfg).set_state(truth['state'])
    require(cfg['episodeSteps']-1-t<=12,'truncated branch horizon')
    feature,counts,hist=origin
    agent._FEAT=deepcopy(feature);agent._COUNTS=deepcopy(counts)
    past=[dict(hist[0]),dict(hist[1])]; plan=None;rows=[];flow=[Counter(),Counter()];dead=[set(),set()]
    gross_deposits=[Counter(),Counter()]; markers=TaggedCargo(uid,seat,sim.workers[uid].carbon); initial=list(sim.cash)
    friendly_conflicts=[];enemy_collisions=[];line='root'
    while sim.step_count<cfg['episodeSteps']-1:
        now=sim.step_count; visible=[sim.obs(p) for p in (0,1)]; own_obs=visible[seat]
        # v1 has hidden feature state. Reuse outputs only for the identical complete public-observation prefix.
        line=digest([line,own_obs]); cached=cache.get(line)
        if cached is None:
            base=clean(agent.agent(own_obs,cfg));cache[line]=(base,deepcopy(agent._FEAT),deepcopy(agent._COUNTS))
        else:
            base,saved_feat,saved_counts=cached;agent._FEAT=deepcopy(saved_feat);agent._COUNTS=deepcopy(saved_counts)
        change=(now==t and role!='v1') or (role=='dynamic' and plan is not None)
        if change:
            selected,plan,meta=policy.choose(own_obs,past[seat],base,cfg,plan=plan,allow_start=now==t)
            require(meta['target_worker']==uid and set(k for k in base.keys()|selected.keys() if base.get(k)!=selected.get(k))<={uid},'secondary worker changed')
        else:
            selected=dict(base); meta={'status':'v1','stage':'no_active_mission','target_worker':None}
        if now==t:
            require(base==clean(truth['joint'][seat])==clean(root['baseline']),'actual root parent mismatch')
            if role!='v1':require(clean(selected)==clean(root['candidate']) and meta==root['meta'] and plan==root['plan'],'public root selection drift')
            opposition=truth['joint'][enemy]
        else:
            opposition=rival.orders(visible[enemy],past[enemy],cfg,variant)
        if role=='single':plan=None
        before=sim.full_observation();counters=sim.worker_counters
        joint=[None,None];joint[seat]=dict(selected);joint[enemy]=dict(opposition)
        for p in (0,1):require(set(joint[p])<=set(visible[p]['players'][p][1])|set(visible[p]['players'][p][2]),'order for unavailable unit')
        result=REF.twin_step(sim,joint); tagged=markers.step(sim,before,counters,joint,result)
        if now==t and role=='v1':
            require(result['score']==truth['step_accounts'][0]['score'] and result['flow']==truth['step_accounts'][0]['flow'],'root first turn vs factual')
        for p in (0,1):
            flow[p].update(result['flow'][p]);dead[p].update(result['deaths'][p]);past[p][now]=visible[p]
            past[p].pop(now-12,None)
        for kind,p,value,*extra in result['events']:
            if kind in ('own_delivery','opponent_worker_delivery_to_this_base','base_collision_cash'):
                gross_deposits[p][extra[0]]+=value
        for e in tagged['focal_collisions']:
            if e['winner']==uid and e['victim_seat']==seat or (e['victim']==uid and any(w.startswith(f'player-{seat}-') and w!=uid for w in e['occupants'])):
                friendly_conflicts.append(dict(pre_t=now,**e))
            elif any(w.startswith(f'player-{enemy}-') for w in e['occupants']):enemy_collisions.append(dict(pre_t=now,**e))
        rows.append({'t':now,'v1':base,'joint':joint,'public_hash_by_seat':[digest(x) for x in visible],'meta':meta,'plan_next':deepcopy(plan),
                     'opponent_order_source':'simultaneous_native_root' if now==t else 'public_branch_proxy',
                     'observation_prefix_sha256':line,'focal':tagged,'cash':result})
        if result['status']==['DONE','DONE']:break
    end=rows[-1]['cash'];require(end['status']==['DONE','DONE'],'did not reach official terminal')
    require(all(abs(initial[p]+sum(flow[p].values())-sim.cash[p])<1e-6 for p in (0,1)),'branch team cash conservation')
    return {'variant':variant,'role':role,'root_t':t,'end_t':sim.step_count,'steps':rows,'official_score':end['score'],
            'end_cash':end['cash'],'margin':end['score'][seat]-end['score'][enemy], 'utility':benefit(end['score'],seat),
            'wdl':(1 if end['score'][seat]>end['score'][enemy] else -1 if end['score'][seat]<end['score'][enemy] else 0),
            'tagged_cargo':markers.summary(),'deaths':[sorted(v) for v in dead],'own_focal_friendly_collisions':friendly_conflicts,
            'own_focal_enemy_collisions':enemy_collisions,'team_flow_by_seat':[dict(x) for x in flow],
            'bank_receipts_by_seat_uid':[dict(x) for x in gross_deposits], 'other_workers_own_bank':sum(v for k,v in gross_deposits[seat].items() if k!=uid),
            'v1_emission_or_exact_public_prefix_reuse':len(rows)}


def diffs(old,new,uid,seat):
    def delta(k):return round(new[k]-old[k],8)
    def tagged(tag,k):return round(new['tagged_cargo'][tag][k]-old['tagged_cargo'][tag][k],8)
    allflows=[]
    for p in (0,1):
        a,b=old['team_flow_by_seat'][p],new['team_flow_by_seat'][p]
        allflows.append({k:round(b.get(k,0)-a.get(k,0),8) for k in sorted(a.keys()|b.keys())})
    return {'margin':delta('margin'),'utility':delta('utility'),'wdl':new['wdl']-old['wdl'],
            'score_change_physical':[round(b-a,8) for a,b in zip(old['official_score'],new['official_score'])],
            'cash_change_physical':[round(b-a,8) for a,b in zip(old['end_cash'],new['end_cash'])],
            'principal_banked':tagged('principal','own_bank'),'fresh_harvested':tagged('fresh','issued'),
            'fresh_banked':tagged('fresh','own_bank'),'fresh_inventory':tagged('fresh','inventory_at_end'),
            'other_workers_bank':delta('other_workers_own_bank'),'team_flow_delta':allflows,
            'added_deaths_by_physical_seat':[sorted(set(new['deaths'][p])-set(old['deaths'][p])) for p in (0,1)],
            'avoided_deaths_by_physical_seat':[sorted(set(old['deaths'][p])-set(new['deaths'][p])) for p in (0,1)]}


def evaluate_game(agent,game,root,bindings):
    key=game['seed'],game['seat'],root['t']; truth=data.audit_root(key)
    # Deliberately discard the entire stored future immediately. Only the two simultaneous root orders are consumed.
    del truth['remaining_joint'];truth['step_accounts']=truth['step_accounts'][:1]
    origin=warm_start(agent,game,root,truth); uid=root['meta']['target_worker']; outputs=[]
    cache={}
    for variant in range(3):
        arms={k:track_arm(agent,origin,truth,root,k,variant,cache) for k in ('v1','single','dynamic')}
        match=arms['single']['steps'][0]; dyn=arms['dynamic']['steps'][0]
        require(match['joint']==dyn['joint'] and match['cash']==dyn['cash'],'candidate root arms diverged too early')
        outputs.append({'variant':variant,'arms':arms,'vs_v1':{k:diffs(arms['v1'],arms[k],uid,game['seat']) for k in ('single','dynamic')},
                        'dynamic_vs_single':diffs(arms['single'],arms['dynamic'],uid,game['seat'])})
    return {'bindings':bindings,'game':game,'root':root,'original_native_terminal_for_context':truth['original'],
            'distinct_public_v1_prefixes_evaluated':len(cache),'scenarios':outputs}


def summary(games, records):
    out={}
    for split in ('dev','eval'):
        sample=[g for g in games if g['split']==split]; ran=[r for r in records if r['game']['split']==split]
        by_key={(r['game']['seed'],r['game']['seat']):r for r in ran}; parts={}
        for v in range(3):
            parts[v]={}
            for mode in ('single','dynamic','dynamic_vs_single'):
                rows=[]
                for g in sample:
                    r=by_key.get((g['seed'],g['seat'])); diff={}
                    if r:
                        scenario=r['scenarios'][v];diff=scenario['dynamic_vs_single'] if mode=='dynamic_vs_single' else scenario['vs_v1'][mode]
                    rows.append({'seed':g['seed'],'seat':g['seat'],'terrain':g['terrain_sha256'],'issued':bool(r), **{
                        k:diff.get(k,0.) for k in ('margin','utility','wdl','fresh_harvested','fresh_banked','principal_banked','other_workers_bank','fresh_inventory')}})
                grouped={terrain:[r for r in rows if r['terrain']==terrain] for terrain in {g['terrain_sha256'] for g in sample}}
                require(len(grouped)==4 and all({r['seat'] for r in rr}=={0,1} for rr in grouped.values()),'incomplete cohort')
                metric={k:statistics.mean(statistics.mean(r[k] for r in rr) for rr in grouped.values()) for k in
                        ('margin','utility','wdl','fresh_harvested','fresh_banked','principal_banked','other_workers_bank','fresh_inventory')}
                parts[v][mode]={'all_map_equal_mean':metric,'all_game_contrasts':rows,'min_margin':min(r['margin'] for r in rows),
                    'positive_zero_negative':[sum(r['margin']>1e-6 for r in rows),sum(abs(r['margin'])<=1e-6 for r in rows),sum(r['margin']< -1e-6 for r in rows)],
                    'wins_by_arm_on_intervened_games': {name:sum(r['scenarios'][v]['arms'][name]['wdl']==1 for r in ran) for name in ('v1','single','dynamic')}}
        changed_maps=len({r['game']['terrain_sha256'] for r in ran}); med=statistics.median(parts[v]['dynamic']['all_map_equal_mean']['margin'] for v in parts)
        lowest=min(parts[v]['dynamic']['min_margin'] for v in parts)
        out[split]={'all_games':len(sample),'all_maps':4,'issued_games':len(ran),'issued_maps':changed_maps,'variants':parts,
                    'median_variant_equal_map_dynamic_margin':med,'worst_dynamic_root_variant':lowest,
                    'registered_conditions':{'at_least_two_terrains':changed_maps>=2, 'positive_median_margin':med>0, 'no_loss_over_100':lowest>=-100}}
    return out


def main():
    import os
    bindings,games=inputs(); matrix,selected=select(); export=HERE/'branches';export.mkdir(exist_ok=True)
    record_path=HERE/'launch_scan.json.gz'
    write(record_path,{'bindings':bindings,'rows':matrix,'earliest':[s for s in selected.values()]})
    agent=module('corridor_sealed_real_v1',PACK);records=[];seals={}
    for game in games:
        key=game['seed'],game['seat'];root=selected.get(key)
        if root is None:continue
        path=export/f"{key[0]}_{key[1]}.json.gz"
        if path.is_file():
            record=read(path)
            require(record['bindings']==bindings and record['game']==game and record['root']==root and len(record['scenarios'])==3,'stale branch checkpoint')
        else:
            record=evaluate_game(agent,game,root,bindings)
            write(path,record)
        records.append(record);seals[path.name]=sha(path)
        print(f"{game['split']} {key} t{root['t']} {[s['vs_v1']['dynamic']['margin'] for s in record['scenarios']]}",flush=True)
    outcomes=summary(games,records)
    result={'schema':'three-policy-controlled-sapling-continuations-v1','bindings':bindings,'scan_sha256':sha(record_path),
            'scope':'Only the first simultaneous opponent order is genuine. Every later enemy order is generated from that branch and seat by one of three fixed public proxies; their contrasts are dependent sensitivity analyses, not native strength estimates.',
            'arms':'v1 on every branch-specific observation, root edit then v1, or one cancellable mission over v1. No new native callbacks.',
            'cargo_accounting':'Focal root principal and newly harvested carbon are carried separately through friendly/enemy inventory, smelting, tax and physical bank. Ground drops close tracing at drop; no later ground harvest is credited to the lost tagged lot.',
            'zero_firing_seats':[g for g in games if (g['seed'],g['seat']) not in selected],'checkpoints_sha256':seals,'outcomes':outcomes}
    result['eligible_to_request_separate_native_intervention']=all(outcomes['eval']['registered_conditions'].values())
    write(HERE/'evaluation.json',result)
    print({s:(z['issued_games'],z['median_variant_equal_map_dynamic_margin'],z['worst_dynamic_root_variant']) for s,z in outcomes.items()},flush=True)


if __name__=='__main__':
    with (HERE/'.evaluate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
