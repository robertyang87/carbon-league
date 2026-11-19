"""Imitate the opponent's public commands without consuming its current order or private view."""
from collections import Counter
from pathlib import Path
import hashlib
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BRIDGE = HERE.parent
ACTOR = BRIDGE.parent
BASE = ACTOR.parent
ROOT = BASE.parents[1]
sys.path[:0] = [str(BRIDGE), str(BASE)]
import bridge
from tree_win_gate import validate_examiner_configuration

F = bridge.F
ROLES = ('collector', 'planter', 'base')
load = bridge.load
sha = bridge.sha
check = bridge.require

def identity():
    deny_maps, deny_seeds = set(), set()
    barred = {}
    for name in ('first_win_dev', 'first_win_blend_dev'):
        clearance = ACTOR / name / 'clearance.json'
        protocol = ACTOR / name / 'protocol.json'
        physical = load(clearance)['physical_boards']
        seeds = load(protocol)['seeds']
        check(set(map(str, seeds)) == set(physical), 'clearance does not cover registered maps')
        deny_maps.update(physical.values())
        deny_seeds.update(seeds)
        barred[name] = {'n':len(seeds), 'protocol_sha256':sha(protocol), 'clearance_sha256':sha(clearance)}
    records = []
    split = {'fit':set(), 'rollout_hold':set()}
    for r in load(BRIDGE/'fit.json')['trajectories']:
        p = ROOT/r['raw']
        raw = load(p)
        check(sha(p) == r['raw_sha256'], 'opponent tape mismatch')
        f = p.parent/'official_replay.json'
        if f.exists():
            candidates = load(f)['runs'][p.parent.name]['games']
            matches = [v for v in candidates if (v['seed'],v['seat']) == (r['seed'],r['side'])]
            check(len(matches) == 1,'missing or duplicate official certificate')
            cert=matches[0]
        else:
            f=p.parent/f"official_seat{r['side']}.json"
            cert=load(f)['official']
        check(cert['map_sha256'] == r['terrain'] == raw['initial_terrain_u8_sha256']
              and cert['transcript_sha256'] == sha(p) and cert['statuses']==['DONE','DONE']
              and cert['official_reward_by_seat']==raw['official_score_by_seat'], 'official mismatch')
        validate_examiner_configuration(raw['configuration'])
        check(raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors']
              and not raw['audit_messages'] and any(v[1]=='searched' for v in raw['search_calls']),
              'unverified examiner')
        check(r['seed'] not in deny_seeds and r['terrain'] not in deny_maps, 'exposed new map')
        split[r['split']].add(r['terrain'])
        records.append({**r,'official':str(f.relative_to(ROOT)),'official_sha256':sha(f),
                        'steps':cert['steps'],'examiner_search_counts':dict(Counter(v[1] for v in raw['search_calls']))})
    check(len(records)==16 and len(split['fit'])==6 and len(split['rollout_hold'])==2
          and not split['fit']&split['rollout_hold'],'split mismatch')
    return records,barred

def source_frames(r):
    raw=load(ROOT/r['raw'])
    candidate=r['side']
    own=F.Feat(candidate)
    enemy=F.Feat(1-candidate)
    sim=bridge.FastCarbon(raw['game_configuration']).reset(seed=r['seed'])
    for t,pair in enumerate(raw['actions']):
        obs=sim.obs(candidate)
        own.observe(obs)
        x=enemy.observe(bridge.public_enemy(obs,own.est))
        yield t, obs, enemy, x, pair[1-candidate]
        sim.step(*pair)
    check(list(sim.cash)==raw['last_post_cash'], 'source tape did not restore actual final cash')

def training_inputs(encoder, x, old):
    prefix=F.Prefix(x,encoder.aux)
    slots=[]
    for s in np.flatnonzero(x['tok_mask'][:F.OWN]):
        mat, legal=bridge.features(x,prefix,s)
        # Only the old model's own predicted prefix is input to the next opponent slot.
        pred=int(np.argmax(np.where(legal,bridge.predict(old,mat),-np.inf)))
        slots.append((int(s),mat,legal))
        prefix.advance(s,pred)
    return slots

def fit(records,old):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(118063)
    mats,masks,ys=[],[],[]
    counts=Counter()
    for r in records:
        if r['split']!='fit':
            continue
        for _,obs,enemy,x,command in source_frames(r):
            generated=training_inputs(enemy,x,old)
            y=enemy.labels(command)
            check(enemy.to_actions(y)=={k:v for k,v in command.items() if v and v!='NONE'}, 'unsupported source act')
            for s,mat,legal in generated:
                role=ROLES[int(x['tok_kind'][s])]
                counts[role+'/all']+=1
                if not legal[y[s]]:
                    counts[role+'/unrepresentable']+=1
                elif sum(legal)==1:
                    counts[role+'/forced']+=1
                else:
                    counts[role+'/train']+=1
                    mats.append(mat.astype('float16'));masks.append(legal);ys.append(y[s])
    xx=torch.from_numpy(np.stack(mats));legal=torch.from_numpy(np.stack(masks));yy=torch.from_numpy(np.array(ys))
    del mats,masks,ys
    mlp=torch.nn.Sequential(torch.nn.Linear(xx.shape[-1],40),torch.nn.Tanh(),torch.nn.Linear(40,1,bias=False))
    opt=torch.optim.Adam(mlp.parameters(),lr=.002)
    losses=[]
    for epoch in range(32):
        order=torch.randperm(len(xx));total=0.
        for ix in order.split(384):
            logits=mlp(xx[ix].float()).squeeze(-1).masked_fill(~legal[ix],-1e9)
            loss=torch.nn.functional.cross_entropy(logits,yy[ix])
            opt.zero_grad();loss.backward();opt.step()
            total+=float(loss.detach())*len(ix)
        losses.append(total/len(xx))
    wt=dict(hidden=mlp[0].weight.detach().numpy().T,bias=mlp[0].bias.detach().numpy(),head=mlp[2].weight.detach().numpy().T)
    with torch.no_grad():
        err=float(np.max(np.abs(np.stack([bridge.predict(wt,q.numpy()) for q in xx[:120].float()])
                                -mlp(xx[:120].float()).squeeze(-1).numpy())))
    check(err<2e-5,'NumPy/Torch score mismatch')
    np.savez_compressed(HERE/'opponent.npz',**wt)
    return dict(training_slot_counts=dict(counts),epoch_mean_nll=losses,numpy_max_error=err,
                parameters=sum(v.size for v in wt.values()),weights_sha256=sha(HERE/'opponent.npz'))

def counts_for(rows):
    report={}
    for t,slots in rows:
        whole={}
        for k in ('old','new','no_order'):
            hit=0
            for uid,role,tags,truth,prev,after in slots:
                value=prev if k=='old' else after if k=='new' else 'NONE'
                for cell in ('all',role,*tags):
                    bucket=report.setdefault((k,cell),Counter())
                    bucket['slots']+=1; bucket['correct']+=value==truth
                    if truth!='NONE':
                        bucket['active']+=1;bucket['active_correct']+=value==truth
                hit+=value==truth
            whole[k]=(hit==len(slots))
            b=report.setdefault((k,'joint'),Counter());b['slots']+=1;b['correct']+=whole[k]
    return {k:{s:dict(v) for (arm,s),v in report.items() if arm==k} for k in ('old','new','no_order')}

def evaluate(records,old,wt):
    receipts=[]
    worlds=[]
    for r in records:
        if r['split']!='rollout_hold':continue
        rows=[]
        for t,obs,enemy,x,truth in source_frames(r):
            before,_=bridge.decode(enemy,x,old)
            after,_=bridge.decode(enemy,x,wt)
            home=obs['players'][1-r['side']]
            target=obs['players'][r['side']]
            units={**{k:'base' for k in home[1]},**{k:v[2].lower() for k,v in home[2].items()}}
            check(set(truth)|set(before)|set(after)<=set(units),'forecast has foreign token')
            slots=[]
            for uid,role in units.items():
                tags=[]
                if role=='collector':
                    pos,cargo,_=home[2][uid]
                    if cargo>=100:tags.append('loaded_100')
                    if cargo>0 and min(F.DIST[pos,p] for p in home[1].values())<=3: tags.append('loaded_near_home')
                    if any(v[2]=='COLLECTOR' and v[1]>cargo and F.DIST[pos,v[0]]<=2 for v in target[2].values()):
                        tags.append('reachable_heavier_sapling')
                slots.append([uid,role,tags,truth.get(uid,'NONE'),before.get(uid,'NONE'),after.get(uid,'NONE')])
            rows.append([t,slots])
        receipts.append({'terrain':r['terrain'],'seed':r['seed'],'seat':r['side'],'source_sha256':r['raw_sha256'],'rows':rows})
        worlds.extend(rows)
    bridge.save(HERE/'held_orders.json.gz',receipts)
    result={'pooled':counts_for(worlds),'maps':{}}
    for terrain in sorted({g['terrain'] for g in receipts}):
        result['maps'][terrain]=counts_for([v for g in receipts if g['terrain']==terrain for v in g['rows']])
    return result

def main():
    oldfile=BRIDGE/'rollout.npz';old=dict(np.load(oldfile,allow_pickle=False))
    check(sha(oldfile)==load(BRIDGE/'fit.json')['weights_sha256'],'wrong historical reference')
    records,barred=identity()
    manifest={'records':records,'excluded':barred,'protocol_sha256':sha(HERE/'protocol.json'),
              'original_sha256':hashlib.sha256(subprocess.check_output(['git', 'cat-file', 'blob', '9c46f6fa4ff401b05939150054c2f88f549c2f7e'],cwd=ROOT)).hexdigest(),'old_weights_sha256':sha(oldfile),
              'training_source_sha256':sha(Path(__file__)),'feature_source_sha256':sha(ROOT/'code/cnet2_feat.py'),
              'bridge_source_sha256':sha(BRIDGE/'bridge.py')}
    bridge.save(HERE/'sources.json',manifest)
    training=fit(records,old)
    bridge.save(HERE/'fit.json',training)
    print('trained',training['weights_sha256'],flush=True)
    results=evaluate(records,old,dict(np.load(HERE/'opponent.npz',allow_pickle=False)))
    results['held_orders_sha256']=sha(HERE/'held_orders.json.gz')
    results['fit_sha256']=sha(HERE/'fit.json');results['manifest_sha256']=sha(HERE/'sources.json')
    bridge.save(HERE/'evaluation.json',results)
    print('evaluated',results['pooled']['new']['joint'],flush=True)
if __name__=='__main__':
    main()
