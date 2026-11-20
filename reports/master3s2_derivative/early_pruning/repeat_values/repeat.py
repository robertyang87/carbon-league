"""Compare and fit multi-depth neural values against independent native continuations."""
import gzip
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from register import BASE, PARENT, PARENT_SHA, sha, save
from collect import collect, instrument
DRAWS=(0,79031,13019,25997,52627,85121)
DEPTHS=(1,4,8)


def plan():
    first=json.loads((HERE.parent/'protocol.json').read_bytes())
    cards=first['sources'].copy()
    novel=[]
    for p in sorted((BASE/'dev1/hosted').glob('*.zip')):
        with zipfile.ZipFile(p) as archive:
            r=json.loads(archive.read('opponent_policy/registration.json'))
            if r['seed']-621758000 not in [580,581,583] or r['candidate_seat']!=r['seed']%2:continue
            novel.append(dict(archive=str(p.relative_to(BASE)), sha256=sha(p.read_bytes()), role='opponent_policy',
                              seed=r['seed'],seat=r['candidate_seat'],terrain=r['physical_map_sha256'],split='extra'))
    assert len(novel)==3 and not {r['terrain'] for r in novel}.intersection(r['terrain'] for r in cards)
    assert len({r['terrain'] for r in cards+novel})==len(cards)+3
    save(HERE/'protocol.json',dict(parent_sha256=PARENT_SHA,steps=first['steps'],
      sources=cards+sorted(novel,key=lambda r:r['seed']),draws=DRAWS,depths=DEPTHS,
      design='12 training maps; original 4 exploratory maps exposed to previous H1-head diagnostics; three separate additional terrain IDs not used in design/fit. Same 8 original native joint orders per root and physical opponent root command. Conditional comparison only.',
      objective='Fit 3 nonnegative sum-one coefficients on within-root H1/H4/H8, centered native critic scores. Each training draw predicts mean H8 of the other training draws. Weights from simplex grid with .025 spacing; start closest to original on ties. Final rerank only the native same-depth finalists; leave early pruning as original. No posthoc coefficients.',
      validation='For draws 0 and 79031, each predicts mean H8 of four independent later draws, evaluating the real cross-future change in the chosen finalist vs native. Both evaluation draws must have positive equal-map change on the 3 extra maps, positive on >=2 of these 3 terrains, nonnegative on both initially exposed 4-map groups. Requires future different whole-game head-to-head to establish strength.'))


def file_for(card,salt):
    prior=HERE.parent/f"{card['split']}_{salt}"/f"{card['seed']}.json.gz"
    if prior.is_file():return prior
    return HERE/'draws'/f"{card['seed']}_{salt}.json.gz"


def gather(split):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['CARBON_DEBUG']='1'
    if split!='train' and not (HERE/'weights.json').exists():raise ValueError('freeze weights before held labels')
    assert sha(PARENT.read_bytes())==PARENT_SHA
    protocol=json.loads((HERE/'protocol.json').read_bytes())
    cfg_src=BASE.parent/'sapling_1200k/master6_actor_dev/cargo_tree_action_learning_dev/direct_original/games/candidate/raw_621758346_1.json.gz'
    cfg=json.loads(gzip.decompress(cfg_src.read_bytes()))['game_configuration']
    model=instrument()
    (HERE/'draws').mkdir(exist_ok=True)
    for card in protocol['sources']:
        if card['split']!=split:continue
        for salt in DRAWS:
            dest=file_for(card,salt)
            if dest.exists():continue
            rows=collect(card,protocol,cfg,model,salt)
            for row in rows:row.pop('features',None)
            raw=gzip.compress(json.dumps(rows,allow_nan=False).encode(),mtime=0)
            tmp=dest.with_suffix('.partial')
            tmp.write_bytes(raw);tmp.replace(dest)
        print(card['seed'],flush=True)


def groups(split):
    for card in json.loads((HERE/'protocol.json').read_bytes())['sources']:
        if card['split']!=split:continue
        draws=[json.loads(gzip.decompress(file_for(card,s).read_bytes())) for s in DRAWS]
        for roots in zip(*draws):
            if 'menu' not in roots[0]:
                assert all('menu' not in r for r in roots)
                continue
            assert all(r['menu']==roots[0]['menu'] and r['1']==roots[0]['1'] for r in roots)
            yield card['seed'], roots[0]['step'], np.array([[r[str(d)] for d in DEPTHS] for r in roots])


def fit():
    blocks=[]
    for seed,step,values in groups('train'):
        for i,observed in enumerate(values):
            target=np.delete(values[:,2,:],i,axis=0).mean(0)
            blocks.append((seed,observed.T-observed.T.mean(0),target-target.mean()))
    candidates=sorted([(a/40,b/40,(40-a-b)/40) for a in range(41) for b in range(41-a)],key=lambda w:(-w[2],-w[1]))
    def loss(w):
        bymap={}
        for seed,x,y in blocks:
            bymap.setdefault(seed,[]).append(np.mean((x@w-y)**2))
        return float(np.mean([np.mean(v) for v in bymap.values()]))
    scores=[]
    # Compress the squared loss before enumerating the simplex.
    bymap={}
    for seed,x,y in blocks:
        bymap.setdefault(seed,[]).append((x.T@x/len(y),x.T@y/len(y),np.mean(y*y)))
    G=np.mean([np.mean([g for g,_,_ in b],axis=0) for b in bymap.values()],axis=0)
    h=np.mean([np.mean([h for _,h,_ in b],axis=0) for b in bymap.values()],axis=0)
    k=np.mean([np.mean([k for _,_,k in b]) for b in bymap.values()])
    for w in candidates:scores.append(float(w@G@w-2*h@w+k))
    idx=int(np.argmin(scores)); w=candidates[idx]
    assert abs(loss(w)-scores[idx])<1e-10
    save(HERE/'weights.json',dict(native_depths=DEPTHS,weights=w,train_maps=len(bymap),draw_cases=len(blocks),
         source_sha256=sha((HERE/'protocol.json').read_bytes()),training_mse=scores[idx],native_training_mse=loss((0,0,1))))


def settle():
    w=json.loads((HERE/'weights.json').read_bytes())['weights'];result={}
    for split in ['held','extra']:
        result[split]={}
        for i in (0,1):
            rows=[]
            for seed,step,values in groups(split):
                first,four,deep=values[i]
                truth=values[2:,2,:].mean(0); n=len(first)
                stage1=[0]+sorted(range(1,n),key=lambda j:-first[j])[:max(2,int(np.ceil(n*.5)))-1]
                finals=[0]+sorted(stage1[1:],key=lambda j:-four[j])[:max(2,int(np.ceil(n*.25)))-1]
                scores=np.asarray(w)@values[i]
                base=max(finals,key=lambda j:deep[j]);choice=max(finals,key=lambda j:scores[j])
                rows.append(dict(seed=seed,step=step,finalists=finals,native=base,choice=choice,
                                 gain=float(truth[choice]-truth[base]),targets=[float(truth[j]) for j in finals]))
            groupscores={str(s):float(np.mean([r['gain'] for r in rows if r['seed']==s])) for s in sorted({r['seed'] for r in rows})}
            result[split][str(DRAWS[i])]=dict(map_gain=groupscores,equal_map_gain=float(np.mean(list(groupscores.values()))),
                          changed=sum(r['choice']!=r['native'] for r in rows),roots=len(rows),rows=rows)
    qualified=all(r['equal_map_gain']>=0 for r in result['held'].values()) and all(
        r['equal_map_gain']>0 and sum(x>0 for x in r['map_gain'].values())>=2 for r in result['extra'].values())
    save(HERE/'evaluation.json',dict(qualified_for_real_response=qualified,groups=result))

if __name__=='__main__':
    command=sys.argv[1]
    if command=='register':plan()
    elif command=='gather':gather(sys.argv[2])
    elif command=='fit':fit()
    elif command=='settle':settle()
    else:raise ValueError(command)
