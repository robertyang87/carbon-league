"""Reconstruct the ten public pre-contact decisions preceding the recorded 462/0 root."""
import gzip
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from focal_survival import VECTORS,destination,play

FOLDER=HERE.parent/'outputs/621758462_0'
FOCAL='player-0-worker-6'
ENEMY='player-1-worker-7'


def distance(a,b,w):
    y,x=divmod(a,w);v,u=divmod(b,w)
    return min(abs(y-v),w-abs(y-v))+min(abs(x-u),w-abs(x-u))


def role_orders(obs,commands,w,focus):
    players=obs['players'];out=[]
    for side,p in enumerate(players):
        items={}
        for uid,(pos,load,role) in p[2].items():
            action=commands[side].get(uid,'NONE')
            items[uid]=dict(pos=pos,cargo_as_visible=load,role=role,command=action,
                            destination=destination(pos,action,w),distance_to_focal=distance(pos,focus,w))
        out.append(dict(cash=p[0],bases=p[1],base_orders={u:commands[side].get(u,'NONE') for u in p[1]},
                        workers=items,trees=p[3]))
    return out


def main():
    verified=play.read(FOLDER/'verified.json');survival=play.read(FOLDER/'focal_survival.json')
    p=play.read(FOLDER/'prefix.json');m=p['root_menu'];terrain=p['terrain']
    for name,h in survival['source_sha256'].items():
        play.need(play.sha((FOLDER/name).read_bytes())==h,'Focal audit source changed '+name)
    play.need(verified['focal_event']['focal_uid']==m['focal_uid']==FOCAL and p['root_step']==84 and not verified['eligible'],'Wrong registered root')
    env,cfg,live=play.official(621758462,terrain,0);bank=[5.,5.];w=int(cfg['size'])
    records=[json.loads(b) for b in gzip.open(FOLDER/'prefix.jsonl.gz')]
    turn84={arm:json.loads(gzip.open(FOLDER/(arm+'.jsonl.gz')).readline()) for arm in 'AB'}
    history=[]
    for t,row in enumerate(records+[turn84['A']]):
        public=[live.observe(env,s,bank[s]) for s in (0,1)]
        play.need([play.sha(live.encoded(o)) for o in public]==row['public_before_by_seat'],'Incorrect public state')
        if t>=74:
            focal=public[0]['players'][0][2][FOCAL][0]
            enemy=public[0]['players'][1][2][ENEMY][0]
            snapshots=role_orders(public[0],row['orders'],w,focal)
            focal_options={cmd:destination(focal,cmd,w) for cmd in VECTORS}
            enemy_reach={destination(enemy,a,w) for a in VECTORS}
            history.append(dict(step=t,acting_public_sha256=row['public_before_by_seat'][0],enemy_public_sha256=row['public_before_by_seat'][1],
                                own_public=snapshots,actual_orders_by_seat=row['orders'],
                                focused_distance=distance(focal,enemy,w),public_enemy_reachable_destinations=sorted(enemy_reach),
                                own_intents_within_enemy_one_move=[k for k,cell in focal_options.items() if cell in enemy_reach]))
        if t==84:break
        env.step(row['orders'],[]);bank=row['bank_after']
        play.need(play.board_hash(env,terrain)==row['board_after_sha256'],'Official advancement differs')
    root=history[-1];own,foe=root['own_public']
    play.need(own['workers'][FOCAL]['pos']==129 and foe['workers'][ENEMY]['pos']==130 and
              all(129 not in side['bases'].values() and 129 not in [v[0] for v in side['trees'].values()] for side in (own,foe)),'Wrong terrain objects')
    play.need(turn84['A']['native_request_id']==turn84['B']['native_request_id'] and turn84['A']['orders'][1]==turn84['B']['orders'][1],'Enemy root action disagrees')
    prior=survival['arms'];a,b=prior['A'],prior['B']
    play.need(a['official_focal_after'][0]==114 and b['official_focal_after'] is None and
              b['official_balance_after_by_seat']==a['official_balance_after_by_seat'] and
              all(not v for v in b['trees_on_cell_after_by_seat']),'Wrong root effects')
    inside={x['uid']:x for x in b['nearby_public_workers_and_realized_response'] if x['enters_focal_cell']}
    play.need(set(inside)=={FOCAL,ENEMY} and inside[ENEMY]['role']=='COLLECTOR' and
              inside[ENEMY]['worker_after_step'][0]==129,'Different participants or surviving UID')
    compare={}
    for uid in sorted(set(turn84['A']['orders'][0])|set(turn84['B']['orders'][0])):
        actions=[turn84[x]['orders'][0].get(uid,'NONE') for x in 'AB']
        if actions[0]!=actions[1]:
            pos=own['workers'][uid]['pos']
            compare[uid]=dict(pre_position=pos,actions=actions,destinations=[destination(pos,c,w) for c in actions])
    # Consume the upstream single-step official contact proof, without starting a policy process.
    paths=['prefix.json','verified.json','focal_survival.json']
    sources={k:play.sha((FOLDER/k).read_bytes()) for k in paths}
    sources.update({str(k):v for k,v in survival['source_sha256'].items()})
    result=dict(seed=621758462,seat=0,root_step=84,focal=FOCAL,enemy=ENEMY,
                scope='Two-seat public state and visible turn history. Current simultaneously-issued commands and realized events are separate replay evidence.',
                own_public_turns_74_to_84=history,source_sha256=sources,root_differing_own_workers=compare,
                shared_original_request=turn84['A']['native_request_id'],changed_enemy_workers=0,
                signed_rule_price=m['public_conditions']['public_rule_price'],root_focal_event=verified['focal_event'],
                root_survival=prior,relative_cutoffs={'prior_3':81,'prior_5':79,'prior_10':74},
                programs={'replay.py':play.sha(Path(__file__).read_bytes()),'focal_survival.py':play.sha((HERE.parent/'focal_survival.py').read_bytes()),
                          'official_helpers.py':play.sha((play.ROOT/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/helpers.py').read_bytes())})
    play.save(HERE/'evidence.json',result)
    for frame in history:
        own,foe=frame['own_public'];p=own['workers'][FOCAL];q=foe['workers'][ENEMY]
        print(frame['step'],p['pos'],p['command'],q['pos'],q['command'],frame['focused_distance'],len(own['trees']),len(foe['trees']),[own['cash'],foe['cash']])
    print('changed',compare,'after',a['official_balance_after_by_seat'])


if __name__=='__main__':
    main()
