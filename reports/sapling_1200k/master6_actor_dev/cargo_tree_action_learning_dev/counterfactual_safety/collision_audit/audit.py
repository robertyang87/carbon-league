"""Classify collision cells once; bind every dead ID and cash/ground/winner route to real fixed native orders."""
import collections, gzip, hashlib, json, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SAFETY=HERE.parent
EXP=SAFETY.parent
sys.path.insert(0,str(SAFETY))
from gather import arena,read,sha,write,FastCarbon,CashEvents
from analyze import close

def cards():
    panel=read(EXP/'panel.json')
    rows=[dict(c,cohort='historic58') for c in panel['cards']]
    for area in ['direct_original','replicate_original']:
        folder=EXP/area
        for f in sorted(folder.glob('complete_*.json')):
            c=read(f);g=folder/'games'/c['arm']/f"raw_{c['seed']}_{c['seat']}.json.gz";receipt=g.with_name(g.name.replace('raw_','receipt_').replace('.json.gz','.json'))
            rows.append(dict(path=str(g.relative_to(arena.ROOT)),receipt_path=str(receipt.relative_to(arena.ROOT)),receipt_sha256=c['receipt_sha256'],
                             sha256=c['raw_sha256'],seed=c['seed'],seat=c['seat'],arm=c['arm'],cohort='four_maps16',terrain=read(g)['initial_terrain_u8_sha256']))
    assert len(rows)==74 and len({c['sha256'] for c in rows})==74
    return rows

def outcome(row,reason,credited_to=None,at=None):
    row['trace_end']=reason;row['trace_recipient']=credited_to;row['trace_end_t']=at

def audit(card, mechanism=None):
    raw=read(arena.ROOT/card['path']);cfg=raw['game_configuration']
    assert cfg['smeltCost']==cfg['moveCost']==0 and sha(arena.ROOT/card['path'])==card['sha256'] and sha(arena.ROOT/card['receipt_path'])==card['receipt_sha256']
    assert raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1'
    receipt=read(arena.ROOT/card['receipt_path'])
    if 'official' in receipt:
        official=receipt['official']
    else:
        official=next(g for a in receipt['runs'].values() for g in a['games'] if g.get('transcript_sha256')==card['sha256'])
    assert official['transcript_sha256']==card['sha256'] and official['statuses']==['DONE','DONE'] and close(official['official_reward_by_seat'],raw['official_score_by_seat'])
    sim=FastCarbon(cfg).reset(seed=card['seed']);events=[];past=collections.deque(maxlen=11);carried=collections.defaultdict(list);collisions_checked=0
    for t,factual in enumerate(raw['actions']):
        joint=factual if mechanism is None or t!=mechanism['t'] else [mechanism['own'],mechanism['opponent']]
        before={}
        for p in (0,1):
            for wid in sim.player_workers[p]:
                w=sim.workers[wid];act=joint[p].get(wid,'STAY')
                before[wid]={'uid':wid,'owner':p,'role':'C' if w.is_collector else 'P','p':w.pos,'c':w.carbon,'act':act,
                             'dst':sim._move_nbr[act][w.pos] if act in sim._move_nbr else w.pos,'new':False}
        past.append((t,before.copy()));counts=sim.worker_counters[:]
        terrain=list(sim.carbon);oldtrees={x.pos:(x.age,x.player) for x in sim.trees.values()};bases=dict(sim.base_at)
        sim.cash=CashEvents(sim.cash);sim.step(*joint)
        for k,p,val,*extra in sim.cash.events:
            if k.startswith('hire_'):
                bid,pos=extra;wid=f'player-{p}-worker-{counts[p]}';counts[p]+=1
                before[wid]=dict(uid=wid,owner=p,role='C' if k=='hire_collector' else 'P',p=pos,c=0.,act='RECRUIT',dst=pos,new=True)
        assert counts==sim.worker_counters
        groups=collections.defaultdict(list)
        for w in before.values():groups[w['dst']].append(w)
        deaths={i[0]:i for i in sim.last_collisions};cash={e[3]:e for e in sim.cash.events if e[0] in ('tree_collision_cash','base_collision_cash')}
        deposits={e[3]:e for e in sim.cash.events if e[0] in ('own_delivery','opponent_worker_delivery_to_this_base')}
        expected=set()
        for cell,people in groups.items():
            if len(people)<2:continue
            per=collections.Counter(i['owner'] for i in people)
            category='pure' if len(per)==1 else 'mixed_stack' if max(per.values())>=2 else 'duel'
            cs=[w for w in people if w['role']=='C']
            candidates=[w for w in cs if w['c']==min(q['c'] for q in cs)] if cs else []
            winner=candidates[0] if len(candidates)==1 else None
            start=0. if cell in bases else (cfg['co2FrmWithered'] if cell in oldtrees and oldtrees[cell][0]>=cfg['treeLifespan'] else terrain[cell])
            on_tree=cell in oldtrees and oldtrees[cell][0]<cfg['treeLifespan']
            victims=[]
            # Preserve actual native order of dead units for the 100-carbon cap.
            victim_ids={w['uid'] for w in people if not winner or w['uid']!=winner['uid']}
            ordered=[before[r[0]] for r in sim.last_collisions if r[0] in victim_ids]
            assert len(ordered)==len(victim_ids);expected.update(victim_ids)
            for w in ordered:
                uid=w['uid'];wid,owner,value,recipient,won=deaths[uid];assert wid==uid and owner==w['owner'] and abs(value-w['c'])<1e-6
                part=dict(uid=uid,owner=owner,role=w['role'],cargo=value,destination=None,ground=0.,cap_loss=0.,recipient=recipient)
                victims.append(part)
                incoming=carried.pop(uid,[])
                if winner:
                    assert recipient==won==winner['owner']
                    part['destination']='worker';part['recipient_id']=winner['uid']
                    if value>0:carried[winner['uid']].append(part)
                    carried[winner['uid']].extend(incoming)
                elif on_tree or cell in bases:
                    assert won is None;part['destination']='tree_cash' if on_tree else 'base_cash'
                    assert recipient==(oldtrees[cell][1] if on_tree else bases[cell][0])
                    if value>0:
                        ck=cash[uid];assert ck[0]==('tree_collision_cash' if on_tree else 'base_collision_cash') and ck[1]==recipient and abs(ck[2]-value)<1e-5
                    for p in [part,*incoming]:outcome(p,'cash',recipient,t)
                else:
                    assert recipient is None and won is None
                    gp=min(value,max(0.,cfg['maxCellCarbon']-start));part['destination']='ground';part['ground']=gp;part['cap_loss']=value-gp;start+=gp
                    outcome(part,'loose_or_capped',None,t)
                    for p in incoming:outcome(p,'trace_lost_at_loose_or_cap',None,t)
            prior=[]
            if category!='duel':
                for j in (10,5,3,0):
                    rec=next((roster for stamp,roster in past if stamp==t-j),{})
                    prior.append(dict(dt=-j,workers={i['uid']:rec.get(i['uid']) for i in people}))
            events.append(dict(t=t,cell=cell,category=category,team_counts=dict(per),members=people,winner=winner['uid'] if winner else None,victims=victims,prior=prior))
        assert expected==set(deaths);collisions_checked+=len(expected)
        # Provenance follows a person or an unambiguous later private transfer with zero move/smelt cost.
        # Base settlement precedes harvesting and tree taxes. Stop before any potentially fractional tax.
        for uid,parts in list(carried.items()):
            if uid in deposits:
                e=deposits[uid];assert e[2]>0
                for part in parts:outcome(part,'cash',e[1],t)
                del carried[uid];continue
            w=sim.workers.get(uid)
            if w is None:
                for part in parts:outcome(part,'unobserved_noncollision_removal',None,t)
                del carried[uid];continue
            if any(tree.player!=w.player and (w.pos==tree.pos or w.pos in sim._nbr8[tree.pos]) and tree.age>1 for tree in sim.trees.values()):
                for part in parts:outcome(part,'stopped_at_possible_enemy_tax',None,t)
                del carried[uid]
        if mechanism is not None and t==mechanism['t']:break
    for parts in carried.values():
        for p in parts:outcome(p,'still_carried_at_end',None,t)
    if mechanism is None:
        assert close(sim.cash,raw['official_score_by_seat']) and len(raw['actions'])==299
    return events,dict(**card,steps=t+1,collision_victims_checked=collisions_checked,score=raw['official_score_by_seat'])

def main():
    rows=[];detail=[]
    for c in cards():
        ev,row=audit(c);rows.append(row);detail.append(dict(card=row,events=ev))
    write(HERE/'collisions.json.gz',detail)
    example=read(SAFETY/'near_home_example.json');c=next(c for c in cards() if c['sha256']==example['raw_sha256'])
    examples={}
    for arm in ('candidate','control','parent'):
        ev,_=audit(c,dict(t=296,own=example['all_own_orders'][arm],opponent=example['opponent_order']))
        examples[arm]=[e for e in ev if e['t']==296 and any(w['uid']==example['uid'] for w in e['members'])]
    raw=read(arena.ROOT/c['path'])
    ev,_=audit(c);examples['archived_natural']=[e for e in ev if e['t']==296 and any(w['uid']==example['uid'] for w in e['members'])]
    write(HERE/'t296.json',dict(card=c,examples=examples,archived_own_command=raw['actions'][296][0],unchanged_enemy=example['opponent_order']))
    print(len(rows),sum(x['collision_victims_checked'] for x in rows),flush=True)
if __name__=='__main__':main()
