"""One-step attack and defense test with the opponent's actual archived command."""
import argparse
import ast
import base64
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import zlib
import numpy as np

from interception_fit import HERE, ROOT, intent, own_view, read_game, sha
from interception_layer import choose_actions, MIN_GAIN, CARGO_THRESHOLD, MAX_LOGIT_GAP
import cnet2_feat as F
import cnet2_np as NP
from cnet2_pack import dequantize
from fastsim import FastCarbon
from eval_v53 import CFG
FROZEN = 'dceed7891e9ff25cf9e211e1c9b61d979d27a253b4c0935f2439804372b31389'


def policy(outfile):
    data=outfile.read_bytes()
    if sha(data)!=FROZEN:
        raise ValueError('requires the actual V5 int8 actor')
    constants={}
    for n in ast.parse(data).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            key=n.targets[0].id
            if key in ('_QMETA','_W64','_WMETA'):
                constants[key]=ast.literal_eval(n.value)
    w=dequantize(zlib.decompress(base64.b64decode(constants['_W64'])),constants['_QMETA'])
    w['meta']=np.array(constants['_WMETA'])
    return NP.Policy(w)


def outcome(state, config, actions, enemy_commands, side):
    sim=FastCarbon(config)
    sim.set_state(state)
    joint=[None,None]
    joint[side]=actions
    joint[1-side]=enemy_commands
    sim.step(*joint)
    v=Counter()
    for _, owner, mass, recipient, recipient_id in sim.last_collisions:
        if owner==1-side and mass==0:
            v['opponent_zero_cargo_workers_lost_in_any_collision']+=1
        if owner==1-side and recipient==side:
            v['captured_enemy_cargo']+=mass
            v['captured_with_positive_cargo']+=int(mass>0)
            v['enemy_zero_cargo_removed']+=int(mass==0)
        if owner==side and recipient==1-side:
            v['own_cargo_given_to_enemy']+=mass
        if owner==side and recipient!=side:
            v['own_lost_cargo_all_recipients']+=mass
    v['our_cash']=sim.cash[side]
    v['their_cash']=sim.cash[1-side]
    v['our_carried']=sum(w.carbon for w in sim.workers.values() if w.player==side)
    v['their_carried']=sum(w.carbon for w in sim.workers.values() if w.player!=side)
    v['our_workers']=len(sim.player_workers[side]);v['their_workers']=len(sim.player_workers[1-side])
    v['net_transferred']=v['captured_enemy_cargo']-v['own_cargo_given_to_enemy']
    v['cash_cargo_worker_difference']=(v['our_cash']+v['our_carried']-v['their_cash']-v['their_carried']
                                       +config['recCollectorCost']*(v['our_workers']-v['their_workers']))
    return v


def run_game(model,weights,game_id,config,sim,epochs,seats,step_stride,offset,dev):
    features={side:F.Feat(side) for side in seats}
    previous={side:None for side in seats}
    changed=[]
    roots=0
    contact_roots=0
    recorded_captures=Counter()
    for t,(record,commands,expected) in enumerate(epochs):
        for side in seats:
            obs = sim.obs(side) if dev else own_view(record,side)
            if not dev and sim.full_observation()['players'] != obs['players']:
                raise ValueError(f'logged player state differs before {game_id} step {t}')
            feat=features[side]
            inputs=feat.observe(obs)
            if t%step_stride==offset:
                roots+=1
                me,them=obs['players'][side],obs['players'][1-side]
                contact_roots+=int(any(a[2]=='COLLECTOR' and b[2]=='COLLECTOR'
                                      and max(a[1],b[1])>=CARGO_THRESHOLD and intent.DIST[a[0],b[0]]<=2
                                      for a in me[2].values() for b in them[2].values()))
                base,proposed,decisions=choose_actions(model,feat,inputs,obs,previous[side],config,weights,intent)
                if dev and base!=commands[side]:
                    raise ValueError(f'V5 original command differs on {game_id} step {t}')
                if base!=proposed:
                    if not decisions:
                        raise ValueError('unattributed change')
                    state=sim.get_state()
                    old=outcome(state,config,base,commands[1-side],side)
                    new=outcome(state,config,proposed,commands[1-side],side)
                    delta={k:new[k]-old[k] for k in sorted(set(old)|set(new))}
                    changed.append(dict(step=t,seat=side,decisions=decisions,delta=delta,
                                        baseline_attacker_action=base.get(decisions[0]['uid'],'NONE'),
                                        alternative_attacker_action=proposed.get(decisions[0]['uid'],'NONE'),
                                        baseline=dict(old),alternative=dict(new)))
            previous[side]=obs
        sim.step(*commands)
        if t%step_stride==offset:
            for seat in seats:
                observed=[mass for _,owner,mass,recipient,_ in sim.last_collisions
                          if owner==1-seat and recipient==seat and mass>0]
                recorded_captures['positive_capture_roots']+=int(bool(observed))
                recorded_captures['carbon_transferred']+=sum(observed)
        if expected is not None and sim.full_observation()['players']!=expected:
            raise ValueError(f'logged transition differs for {game_id} at {t}')
    return dict(id=game_id,roots=roots,loaded_contact_roots=contact_roots,actual_original_capture=dict(recorded_captures),changed=changed)


def summarize(games):
    sums=[]
    direct=[]
    for g in games:
        total=Counter()
        for event in g['changed']:
            total.update(event['delta'])
        sums.append(total)
        direct.extend(e['delta'] for e in g['changed'])
    group={k:sum(s[k] for s in sums) for k in sorted(set().union(*sums))}
    net='cash_cargo_worker_difference'
    # All evaluated roots incl. unchanged. Opposite seats from each elite match stay together.
    groups={}
    for g,s in zip(games,sums):
        key=g['id'].split('/')[0]
        value=groups.setdefault(key,[0,0.,0.])
        value[0]+=g['roots'];value[1]+=s[net];value[2]+=s['net_transferred']
    per_map=np.array([v[1]/v[0] for v in groups.values()])
    rng=np.random.default_rng(551308)
    interval=np.quantile(per_map[rng.integers(len(per_map),size=(10000,len(per_map)))].mean(1),[.025,.975])
    return dict(games=len(games),maps=len(groups),roots=sum(g['roots'] for g in games),changed=len(direct),
                real_extra_capture_roots=sum(d.get('captured_enemy_cargo',0)>0 for d in direct),
                real_extra_enemy_empty_kills=sum(d.get('enemy_zero_cargo_removed',0)>0 for d in direct),
                real_prevented_cargo_loss_roots=sum(d.get('own_cargo_given_to_enemy',0)<0 for d in direct),
                changed_without_capture_or_prevented_transfer=sum(d.get('captured_enemy_cargo',0)<=0 and d.get('own_cargo_given_to_enemy',0)>=0 for d in direct),
                maps_with_positive_net_transfer=sum(v[2]>0 for v in groups.values()),
                loaded_contact_roots=sum(g['loaded_contact_roots'] for g in games),
                actual_original_positive_capture_roots=sum(g['actual_original_capture'].get('positive_capture_roots',0) for g in games),
                actual_original_transferred=sum(g['actual_original_capture'].get('carbon_transferred',0) for g in games),
                totals=group,per_map_mean_all_root_economy=float(per_map.mean()),paired_map_bootstrap_95=interval.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--replays',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    fitted=json.loads((HERE/'interception_fit.json').read_text())
    model=policy(ROOT/'data/weights/sapling_1200k/soft_it50_v5_agent.py')
    weights=np.asarray(fitted['weights'])
    index={v['id']:v for v in json.loads(a.manifest.read_text())['games']}
    groups={'top5':[],'v5_original_master3s2':[]}
    for key in fitted['game_ids']['val']:
        item=index[key]
        if item['split']!='val' or key in fitted['game_ids']['train']:
            raise ValueError('data leak')
        archive=read_game(a.replays,item);steps=archive['steps'];config=archive['configuration']
        sim=FastCarbon(config).reset(full_carbon=steps[0][0]['observation']['full_carbon'])
        epochs=((f,[after[side]['action'] or {} for side in (0,1)],after[0]['observation']['players'])
                for f,after in zip(steps,steps[1:]))
        groups['top5'].append(run_game(model,weights,key,config,sim,epochs,(0,1),4,item['seed']%4,False))
    directory=HERE/'tree_exam_soft_it50_v5'
    report=json.loads((directory/'results.json').read_text())
    for row in report['games']:
        blob=(directory/row['raw_file']).read_bytes()
        if sha(blob)!=row['raw_sha256']:raise ValueError('invalid original trajectory')
        raw=json.loads(gzip.decompress(blob))
        if raw['pack_sha256']!=FROZEN or raw['examiner_md5']!='57f9a594cce4f2ec9e3573d8d9d26ca1':
            raise ValueError('wrong actor or examiner')
        sim=FastCarbon(CFG).reset(seed=row['seed'])
        groups['v5_original_master3s2'].append(run_game(model,weights,f"{row['seed']}/{row['candidate_seat']}",CFG,sim,
                    ((None,acts,None) for acts in raw['actions']),[row['candidate_seat']],1,0,True))
    metrics={name:summarize(gs) for name,gs in groups.items()}
    elite=metrics['top5'];dev=metrics['v5_original_master3s2']
    advance=(elite['changed']>=20 and elite['maps_with_positive_net_transfer']>=6
             and elite['totals'].get('net_transferred',0)>0 and elite['paired_map_bootstrap_95'][0]>0
             and dev['totals'].get('net_transferred',0)>0)
    result=dict(schema=1,actor_sha256=FROZEN,thresholds=dict(minimum_expected_capture_or_saved=CARGO_THRESHOLD,
                minimum_estimated_gain=MIN_GAIN,max_policy_logit_gap=MAX_LOGIT_GAP),
                advance_to_two_seat_development=bool(advance),summary=metrics,games=groups,
                sha256={p.name:sha(p.read_bytes()) for p in (Path(__file__),HERE/'interception_layer.py',HERE/'interception_fit.json',a.manifest)})
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
