"""Compare the sole sealed deployed update and its parent on fresh registered replay panels."""
from collections import Counter,defaultdict
from pathlib import Path
import gzip,json
import numpy as np
from train import HERE, PARENT, load, save, sha, assemble, probe, check

def tally(roots):
    z=Counter();r=defaultdict(Counter)
    for x in roots:
        z['roots']+=1
        d=x['demonstration_command']; p=x['arms']['parent']['actual_command']; c=x['arms']['candidate']['actual_command']
        z['joint_changed']+=p!=c;z['parent_joint_matches']+=p==d;z['candidate_joint_matches']+=c==d
        for slot in x['slots'].values():
            a=slot['outputs']['parent'];b=slot['outputs']['candidate'];teacher=slot['demo_canonical']
            row=r[slot['role']];row['occupied']+=1
            row['all_changed']+=a['actual']!=b['actual']
            if min(len(a['legal']),len(b['legal']))<=1:continue
            row['free_both']+=1
            row['parent_agreement']+=teacher==a['actual'];row['candidate_agreement']+=teacher==b['actual']
            row['changed']+=a['actual']!=b['actual']
            row['moved_to_expert']+=a['actual']!=teacher==b['actual']
            row['moved_off_expert']+=a['actual']==teacher!=b['actual']
            row['teachable_in_both_prefixes']+=teacher in a['legal'] and teacher in b['legal']
    return dict(total=dict(z),roles={k:dict(v) for k,v in r.items()})

def by_split(gs):
    game_rows=[dict(id=g['id'],**tally(g['roots'])) for g in gs]; keys=['collector','planter','base']
    macro={};scope={}
    for r in keys:
        samples=[t['roles'][r] for t in game_rows if t['roles'].get(r,{}).get('free_both',0)]
        if not samples:continue
        d=np.asarray([(v['candidate_agreement']-v['parent_agreement'])/v['free_both'] for v in samples]);rng=np.random.default_rng(7924059)
        draws=rng.choice(d,(5000,len(d)),replace=True).mean(1)
        macro[r]=dict(maps=len(d),mean_delta=float(d.mean()),positive=int((d>0).sum()),negative=int((d<0).sum()),same=int((d==0).sum()),descriptive_bootstrap_95=np.quantile(draws,[.025,.975]).tolist())
    for g in gs:
        for root in g['roots']:
            phase=min(root['t']//75,3)
            for k in (f'phase_{phase}',f"teacher_{g['teacher']}",f"opponent_unseen_in_ancestor_{g['opponent_new_to_ancestor']}"):
                scope.setdefault(k,[]).append(root)
    return dict(overall=tally([x for g in gs for x in g['roots']]),macro_terrain_equal=macro,
                by_group={k:tally(roots) for k,roots in scope.items()},maps=game_rows)

def main():
    p=check(); r=load(HERE/'recipe.json');pkg=load(HERE/'package.json')
    assert sha(Path(__file__).read_bytes())==r['evaluator_sha256'] and sha(Path(probe.__file__).read_bytes())==r['probe_sha256']
    assert sha((HERE/'candidate.py').read_bytes())==pkg['candidate_sha256']
    cards=[c for c in assemble() if c['split'] in ('val','holdout')]
    assert len(cards)==len({g['terrain_sha256'] for g in cards})==40
    models={'parent':probe.package(PARENT,'full_parent'),'candidate':probe.package(HERE/'candidate.py','full_final')}
    gs=[]
    for c in cards:
        g=probe.run_game(models,c,p['eval_steps']);g.update({k:c[k] for k in ['seed','teacher','opponent_new_to_ancestor']})
        gs.append(g)
        if len(gs)%5==0:print('serving',len(gs),'of',len(cards),flush=True)
    payload=gzip.compress(json.dumps(gs,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),compresslevel=6,mtime=0)
    dest=HERE/'serving_roots.json.gz'
    with dest.open('xb') as f:f.write(payload)
    results={k:by_split([g for g in gs if g['split']==k]) for k in ('val','holdout')}
    save(HERE/'serving.json',dict(packages=pkg,steps=p['eval_steps'],recipe_sha256=sha((HERE/'recipe.json').read_bytes()),roots_sha256=sha(payload),
         scope='Normal public agent entry, visible demonstrated histories, each arm independently completes its own legal prefix, same return protection. Expert command agreement only. Report all, and occupied slots with >1 legal action under both own prefixes; no tuning or actual strength inference.',results=results))
    for k,v in results.items():print(k,v['overall'],v['macro_terrain_equal'],flush=True)
if __name__=='__main__':main()
