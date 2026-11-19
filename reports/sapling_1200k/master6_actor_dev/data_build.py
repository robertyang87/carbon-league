"""Rebuild sparse teacher-forced roots from the exact physical-seat archive and seal each shard."""
import argparse
from collections import Counter
import gzip,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from pydantic_core import from_json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT/'code'))
import cnet2_feat as F
from cnet2_build import FIELDS

CACHE=Path('/workspace/sapling-all-top5-20261119/replays')
DEFAULT=Path('/workspace/sapling-master6-actor-data')
WINDOWS=((0,75),(75,150),(150,225),(225,299))
PUBLIC=('step','players','trees')
def sha(blob):return hashlib.sha256(blob).hexdigest()
def prep(card,data_root):
    file=data_root/(card['id']+'.npz');sidecar=file.with_suffix('.json')
    recipe=sha(Path(__file__).read_bytes())
    if sidecar.exists():
        info=json.loads(sidecar.read_bytes())
        if info['recipe_sha256']!=recipe or info['source_sha256']!=card['sha256'] or sha(file.read_bytes())!=info['sha256']: raise ValueError('stale cached training root')
        return info
    packed=(CACHE/(card['id']+'.json.gz')).read_bytes()
    if sha(packed)!=card['sha256']:raise ValueError('source replay hash')
    game=from_json(gzip.decompress(packed));frames=game['steps'];cfg=game['configuration'];T=card['turns'];seat=card['seat']
    if len(frames)!=T+1 or sha(bytes(frames[0][0]['observation']['full_carbon']))!=card['terrain_sha256'] or cfg['randomSeed']!=card['seed']:raise ValueError('source map mismatch')
    if list(game['rewards'])!=[frames[-1][s]['reward'] for s in (0,1)] or game['rewards'][seat]-game['rewards'][1-seat]!=card['margin'] and abs(game['rewards'][seat]-game['rewards'][1-seat]-card['margin'])>1.e-6:raise ValueError('result mismatch')
    rng=np.random.default_rng(int(sha(('master6-actor-window-1|'+card['id']).encode())[:16],16))
    selected=set()
    for lo,hi in WINDOWS:
        slot=list(range(lo,min(hi,T)))
        if slot:selected.update(map(int,rng.choice(slot,size=min(8,len(slot)),replace=False)))
    data={k:[] for k in FIELDS};rejected=[];phases=Counter();actions=Counter();n_valid=Counter();f=F.Feat(seat)
    for t,frame in enumerate(frames[:-1]):
        if frame[seat]['status']!='ACTIVE':raise ValueError('non-active decision')
        if t>max(selected,default=-1):break
        shared=frame[0]['observation'];obs={k:shared[k] for k in PUBLIC}
        obs.update(player=seat,carbon=frame[seat]['observation']['carbon'])
        if obs['step']!=t:raise ValueError('timestamp mismatch')
        x=f.observe(obs)
        if t not in selected:continue
        recorded=frames[t+1][seat]['action'] or {}
        expected={k:v for k,v in recorded.items() if v not in (None,'NONE')};y=f.labels(recorded)
        if f.to_actions(y)!=expected:
            rejected.append([t,'unrepresentable command or inactive/foreign ID']);continue
        prefix=F.Prefix(x,f.aux);uf=np.zeros((F.OWN,F.UF),np.float32);mf=np.zeros((F.OWN,5,F.MF),np.float32)
        legal=np.zeros((F.OWN,5),bool);valid=True
        for s in range(F.OWN):
            if not x['tok_mask'][s]:continue
            uf[s],mf[s],legal[s]=prefix.feats(s)
            if not legal[s,int(y[s])]:
                rejected.append([t,'recorded command illegal in live decoder prefix',int(s),int(y[s])]);valid=False;break
            prefix.advance(s,int(y[s]))
        if not valid:continue
        row={k:x[k] for k in FIELDS if k in x}
        # carbon is an auxiliary target retained for loader compatibility. It is never passed
        # to Feat.observe or to the action network. This training run uses policy loss only.
        row.update(uf=uf,mf=mf,legal=legal,y=y,carbon=F.q8(np.array(shared['full_carbon'])[F.CANON if seat else F.GRID]/100.),
                   value=np.tanh(card['margin']/3000.),teacher=2,gid=card['gid'],step=t)
        if row.keys()!=FIELDS.keys():raise ValueError('unsupported schema or environment')
        for k in row:data[k].append(row[k])
        phases[t//75]+=1
        for s in np.flatnonzero(y>=0):
            role='base' if s>=10 else 'collector' if x['tok_kind'][s]==0 else 'planter'
            actions[role+'/'+str(int(y[s]))]+=1;n_valid[role]+=int(legal[s].sum()>1)
    if not data['step']:raise ValueError(f'no valid roots for {card["id"]}')
    arr={k:np.asarray(v,dtype=FIELDS[k][0]) for k,v in data.items()}
    for k,v in arr.items():
        if v.shape!=(len(data['step']),*tuple(FIELDS[k][1])) or not np.isfinite(v).all(): raise ValueError('invalid feature '+k)
    tmp=file.with_suffix('.partial')
    with tmp.open('wb') as handle:np.savez_compressed(handle,**arr)
    os.replace(tmp,file)
    info={'id':card['id'],'gid':card['gid'],'seat':seat,'split':card['split'],'source_sha256':card['sha256'],'recipe_sha256':recipe,
          'file':file.name,'sha256':sha(file.read_bytes()),'rows':len(arr['step']),'steps':data['step'],'selected_before_legality':sorted(selected),'rejected':rejected,
          'phase_roots':dict(phases),'commands':dict(actions),'multiple_choices_by_role':dict(n_valid)}
    tmp=sidecar.with_suffix('.partial');tmp.write_text(json.dumps(info,indent=1)+'\n');os.replace(tmp,sidecar)
    return info

def main(out):
    if not out.is_dir():out.mkdir(parents=True)
    path=HERE/'data_split.json';d=json.loads(path.read_bytes()); cards=[r for r in d['games'] if r['split']!='excluded']
    # Publish training shards first. Within each arm preserve the original frozen, outcome-independent order.
    cards.sort(key=lambda z:['train','val','holdout'].index(z['split']))
    receipts=[]
    for i,g in enumerate(cards):
        rec=prep(g,out);receipts.append(rec)
        if i%12==0: print(i+1,'/',len(cards),g['split'],g['id'],flush=True)
    counts={}
    for split in ('train','val','holdout'):
        rows=[r for r in receipts if r['split']==split];cmds=Counter();phase=Counter();choices=Counter()
        for r in rows:cmds.update(r['commands']);phase.update({str(k):v for k,v in r['phase_roots'].items()});choices.update(r['multiple_choices_by_role'])
        counts[split]={'maps':len(rows),'roots':sum(r['rows'] for r in rows),'attempts':sum(len(r['selected_before_legality']) for r in rows),'phase_roots':phase,'actions':cmds,'multi_choice':choices}
    doc={'split_sha256':sha(path.read_bytes()),'recipe_sha256':sha(Path(__file__).read_bytes()),'canonical_feat_sha256':sha((ROOT/'code/cnet2_feat.py').read_bytes()),
         'canonical_builder_sha256':sha((ROOT/'code/cnet2_build.py').read_bytes()),'weights':{'style_slot':2},'shard_directory':str(out),'counts':counts,'games':receipts}
    full=json.dumps(doc,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode();final=HERE/'data_receipts.json.gz'
    tmp=final.with_suffix('.partial');tmp.write_bytes(gzip.compress(full,mtime=0));os.replace(tmp,final);print(counts,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT);main(p.parse_args().out)
