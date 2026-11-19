"""Reconstruct each examiner's complete visible history and align each clock receipt to its final command."""
from pathlib import Path
import sys,gzip,json,hashlib,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import builder as B
F=B.F
ROOT=B.ROOT
ACTOR=HERE.parents[1]

def read(path):
    raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_cards(plan):
    out=[]
    for fold,tag in (('fit','fit_seeds'),('held','held_seeds'),('diagnostic','diagnostic_seeds')):
        for seed in plan[tag]:
            base=(ACTOR/'broader_elite_backbone_dev/pure_native_gate/games' if fold=='diagnostic' else ACTOR/'first_win_dev/games')
            for arm in (('parent','candidate') if fold=='diagnostic' else ('candidate',)):
                for sapling_seat in (0,1):
                    p=base/arm/f'raw_{seed}_{sapling_seat}.json.gz'
                    win=p.with_name(p.name.replace('raw_','wall_'));proof=p.with_name(p.name.replace('raw_','receipt_').replace('.json.gz','.json'))
                    raw=read(p);wall=read(win);receipt=read(proof)
                    assert sha(p)==receipt['game']['raw_sha256']==wall['raw_sha256']
                    assert raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors'] and not any(raw['agent_errors_by_seat']) if 'agent_errors_by_seat' in raw else (raw['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and not raw['errors'])
                    assert raw['seed']==seed and raw['candidate_seat']==sapling_seat
                    terrain=raw['initial_terrain_u8_sha256'];ident=wall['identity'];assert (ident['terrain'],ident['seat'],ident['seed'],ident['arm'])==(terrain,sapling_seat,seed,arm)
                    assert len(raw['actions'])==len(wall['calls'])==len(set(c['step'] for c in wall['calls']))
                    out.append(dict(gid=len(out),source=str(p.relative_to(ROOT)),source_sha256=sha(p),wall=str(win.relative_to(ROOT)),wall_sha256=sha(win),
                                    receipt=str(proof.relative_to(ROOT)),receipt_sha256=sha(proof),split=fold,seed=seed,terrain=terrain,domain='original',seat=1-sapling_seat,steps=len(raw['actions']),arm=arm))
    maps={fold:{c['terrain'] for c in out if c['split']==fold} for fold in ('fit','held','diagnostic')}
    assert not (maps['fit']&maps['held'] or maps['fit']&maps['diagnostic'] or maps['diagnostic']&maps['held'])
    return out
def make(card,m,path):
    sim,tape=B.replay(card);calls=read(ROOT/card['wall'])['calls'];s=card['seat'];feat=F.Feat(s);tr=m.BeliefTracker(m._Bel,s);previous=None;rows=[];repaired_count=0
    for t,(pair,clock) in enumerate(zip(tape,calls,strict=True)):
        assert t==sim.step_count==clock['step'] and isinstance(clock['changed'],bool)
        assert not clock['changed'] or clock['kind']=='searched'
        obs={k:v for k,v in sim.obs(s).items() if k in B.PUBLIC}
        est,known=tr.step(obs['players'],t,obs['carbon']);x=feat.observe(obs)
        encoded=np.clip(np.rint(m.encode(obs['players'],t,s,est,known,previous)*100),0,255)*.01
        heads=m._POLICY.forward(encoded,m._SID);raw=B.physical_to_student(m,feat,x,obs,heads)
        actual=feat.labels(pair[s]);assert feat.to_actions(actual)=={k:v for k,v in pair[s].items() if v and v!='NONE'}
        r=B.projected(feat,x,raw,actual,0,card['gid'],t)
        invalid=np.flatnonzero(r['prefix_repaired']);cut=int(invalid[0]) if invalid.size else F.OWN
        r['eligible']=(np.arange(F.OWN)<cut)&(actual>=0)&(r['legal'].sum(1)>1)
        if invalid.size:repaired_count+=1
        r['team_changed']=np.bool_(clock['changed'])
        r['search_depth']=np.int8(clock['max_depth'])
        r['source_uid_number']=np.asarray([int(uid.rsplit('-',1)[1]) if uid else -1 for uid in feat.own_uid],np.int16)
        rows.append(r);tr.set_actions(obs['players'],pair[s]);previous=obs['players'];sim.step(*pair)
    tmp=path.with_suffix('.partial')
    with tmp.open('wb') as f:np.savez_compressed(f,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
    tmp.replace(path)
    return dict(steps=len(rows),changed=sum(int(r['team_changed']) for r in rows),usable=sum(int(r['eligible'].sum()) for r in rows),repaired_prefix_roots=repaired_count)
def run():
    plan=read(HERE/'protocol.json');cards=source_cards(plan);m=B.original();dst=HERE/'shards';dst.mkdir(exist_ok=True)
    seal=dict(builder=sha(Path(B.__file__)),local_builder=sha(Path(__file__)),original=B.ORIGINAL_SHA,protocol=sha(HERE/'protocol.json'))
    for c in cards:
        start=time.monotonic();p=dst/f"{c['gid']:02}_{c['seed']}_{c['seat']}_{c['arm']}.npz";rec=p.with_suffix('.json')
        if rec.exists():
            proof=read(rec);assert proof['seal']==seal and proof['card']==c and proof['sha256']==sha(p)
        else:
            result=make(c,m,p);proof=dict(seal=seal,card=c,sha256=sha(p),summary=result);B.save(rec,proof)
        c.update(file=str(p.relative_to(HERE)),sha256=proof['sha256'],**proof['summary'])
        print(c['gid'],c['split'],c['steps'],c['changed'],'%.1fs'%(time.monotonic()-start),flush=True)
    assert sum(c['changed'] for c in cards if c['split']=='diagnostic')==733
    B.save(HERE/'manifest.json',dict(seal=seal,cards=cards))
if __name__=='__main__':run()
