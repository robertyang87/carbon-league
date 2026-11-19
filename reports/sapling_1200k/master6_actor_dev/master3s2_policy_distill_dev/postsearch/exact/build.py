"""Recover Master3s2's actual sampled pre-search commands by substituting the archived post-search return."""
from pathlib import Path
import sys,gzip,json,hashlib,time,os
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import dataset as v1
B=v1.B;F=B.F;ACTOR=v1.ACTOR;ROOT=v1.ROOT;sha=v1.sha;read=v1.read
def clean(a):return {k:v for k,v in a.items() if v and v!='NONE'}

def bundles(p):
    out=[]
    for pattern in p['integer_rng']:
        paths=sorted(ACTOR.glob(pattern));assert len(paths)==2
        for path in paths:
            r=read(path);c=r['case'];assert r['binding']['original']==B.ORIGINAL_SHA and r['binding']['source']==c['source_sha256']
            out.append(dict(mt=None,rec=path,path=ROOT/c['source'],source_sha=c['source_sha256'],t=c['t'],seed=c['seed'],seat=1-c['seat'],scores=r['score'],calls=r['calls'],tape=r['tape'],rng=int(r['initial_rng']),rng_source=path,
                            arm=r['arm'],score_status=r['statuses']))
    for directory in p['mt_state']:
        base=ACTOR/directory;q=read(base/'live_protocol.json');shared=base/'live_shared_init.json'
        for arm in ('parent','approach'):
            path=base/f'live_{arm}.json';r=read(path);src=ROOT/q['record']['source'];wall=base/f'clock_{arm}.json.gz';act=base/r['action_tape']
            assert r['source_sha256']==B.ORIGINAL_SHA and r['shared_init_sha256']==sha(shared) and r['protocol_sha256']==sha(base/'live_protocol.json')
            assert r['action_sha256']==sha(act)==read(wall)['action_sha256'] and r['original_trajectory_sha256']==q['record']['source_sha256']
            # A resumed arm may have reconstructed source sampling state using a different historical implementation; validate it exactly just the same.
            out.append(dict(mt=read(shared),rec=path,path=src,source_sha=q['record']['source_sha256'],t=q['root_t'],seed=q['seed'],seat=1-q['seat'],scores=r['score'],
                            calls=read(wall)['calls'],tape=read(act),rng=None,rng_source=shared,wall=wall,act=act,arm=arm,score_status=r['statuses']))
    assert all(len(c['tape'])==len(c['calls'])+1 and c['score_status']==['DONE','DONE'] for c in out)
    return out

def reconstruct(card,m):
    path=card['path'];assert sha(path)==card['source_sha'];raw=read(path);s=card['seat'];t0=card['t'];cfg=raw['game_configuration']
    assert raw['seed']==card['seed'] and raw['candidate_seat']==1-s and not raw['errors']
    sim=B.FastCarbon(cfg).reset(seed=raw['seed']);terrain=B.sha(np.asarray(sim.carbon,np.uint8).tobytes())
    assert terrain==raw['initial_terrain_u8_sha256']
    feat,tr,prev=F.Feat(s),m.BeliefTracker(m._Bel,s),None
    for t,pair in enumerate([*raw['actions'][:t0],card['tape'][0]]):
        obs=sim.obs(s);feat.observe(obs)
        tr.step(obs['players'],t,obs['carbon']);tr.set_actions(obs['players'],pair[s]);prev=obs['players']
        if t==t0:assert clean(pair[s])==clean(raw['actions'][t][s])
        sim.step(*pair)
    rng=np.random.RandomState(card['rng']);mt=card['mt']
    if mt:rng.set_state((mt['mt_algorithm'],np.asarray(mt['mt_keys'],np.uint32),mt['mt_index'],mt['has_normal'],mt['stored_normal']))
    m._ST.clear();m._ST.update(tr=tr,prev=prev,rng=rng)
    captured={};rows=[]
    def logged_search(obs,cfg,est,sampled,worker_logits,base_logits,prev,t0,home):
        assert not captured
        captured.update(sampled=clean(sampled),heads=(worker_logits,base_logits))
        return clean(pair[s])
    old=m._search;m._search=logged_search
    try:
        for clock,pair in zip(card['calls'],card['tape'][1:],strict=True):
            obs=sim.obs(s);t=sim.step_count;assert clock['step']==t
            x=feat.observe(obs);captured.clear()
            submitted=clean(m._act(obs,cfg,0.0));assert submitted==clean(pair[s])
            pre=captured['sampled'];changed=(pre!=submitted)
            if changed!=clock['changed']:raise ValueError((card['rec'],t,'changed mismatch',pre,submitted,clock))
            actual=feat.labels(submitted);sampled=feat.labels(pre)
            assert feat.to_actions(sampled)==pre and feat.to_actions(actual)==submitted
            rawq=B.physical_to_student(m,feat,x,obs,captured['heads'])
            r=B.projected(feat,x,rawq,actual,0,card['gid'],t);bad=np.flatnonzero(r['prefix_repaired']);cut=int(bad[0]) if bad.size else F.OWN
            r['eligible']=(np.arange(F.OWN)<cut)&(actual>=0)&(r['legal'].sum(1)>1)
            r['unit_changed']=actual!=sampled;assert bool(r['unit_changed'].any())==changed
            r['team_changed']=np.bool_(changed);r['pre_act']=sampled.astype(np.int8);r['source_uid_number']=np.asarray([int(u.rsplit('-',1)[1]) if u else -1 for u in feat.own_uid],np.int16)
            rows.append(r);sim.step(*pair)
    finally:m._search=old
    assert np.allclose(sim.cash,card['scores'],rtol=0,atol=1e-5)
    return rows,terrain
def run():
    os.environ['CARBON_DEBUG']='1';p=read(HERE/'protocol.json');allcards=bundles(p);m=B.original();result=[]
    dest=HERE/'shards';dest.mkdir(exist_ok=True)
    for i,c in enumerate(allcards):
        c['gid']=i;rows,terrain=reconstruct(c,m);f=dest/f"{i:02}_{c['seed']}_{c['arm']}.npz"
        with f.with_suffix('.partial').open('wb') as st:np.savez_compressed(st,**{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        f.with_suffix('.partial').replace(f)
        rec=dict(gid=i,seed=c['seed'],terrain=terrain,seat=c['seat'],arm=c['arm'],first_step=c['t']+1,last_step=int(rows[-1]['step']),split='fit' if c['seed'] in p['fit'] else 'held',
            steps=len(rows),changed=sum(int(r['team_changed']) for r in rows),changed_units=sum(int(r['unit_changed'].sum()) for r in rows),
            trainable_changes=sum(int((r['unit_changed']&r['eligible']).sum()) for r in rows),repaired=sum(int(r['prefix_repaired'].any()) for r in rows),
            source=str(c['path'].relative_to(ROOT)),source_sha256=c['source_sha'],rng_source=str(c['rng_source'].relative_to(ROOT)),rng_source_sha256=sha(c['rng_source']),
            response=str(c['rec'].relative_to(ROOT)),response_sha256=sha(c['rec']),file=str(f.relative_to(HERE)),sha256=sha(f))
        if 'wall' in c:rec.update(clock=str(c['wall'].relative_to(ROOT)),clock_sha256=sha(c['wall']),tape=str(c['act'].relative_to(ROOT)),tape_sha256=sha(c['act']))
        result.append(rec);print(i,rec['seed'],rec['steps'],rec['changed'],rec['changed_units'],flush=True)
    assert {c['seed'] for c in result}==set(p['fit']+p['held']) and not {c['terrain'] for c in result if c['split']=='held'}&{c['terrain'] for c in result if c['split']=='fit'}
    B.save(HERE/'manifest.json',dict(protocol=sha(HERE/'protocol.json'),builder=sha(Path(__file__)),master=B.ORIGINAL_SHA,cards=result))
if __name__=='__main__':run()
