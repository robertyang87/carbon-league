"""Direct paired terminal-U optimization of Sapling's contextual decoder on the deployed mix."""
import argparse,copy,gzip,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from collections import Counter
import numpy as np
from register import H,A,root,digest

sys.path.insert(0,str(A/'blended_actor_dev/serving'))
import build as pack
def imported(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
W=imported('complete_episode_private_world',A/'blended_policy_return_dev/world.py')
read=lambda p:json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def put(p,d):
    raw=(json.dumps(d,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode()
    if p.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    p.parent.mkdir(exist_ok=True,parents=True)
    if p.exists():
        assert p.read_bytes()==raw;return
    tmp=p.with_suffix(p.suffix+'.part');tmp.write_bytes(raw);tmp.replace(p)
P=read(H/'protocol.json'); reg=read(H/'register.json')
assert reg['protocol']==digest(H/'protocol.json')
assets={k:root/P[k]['path'] for k in ('parent','opponent','student_start')}
assert all(digest(v)==P[k]['sha256'] for k,v in assets.items())
CACHE=Path('/workspace/sapling-it50-full-episode-temporary');CACHE.mkdir(exist_ok=True)
def check(seed):
    assert seed in P['seeds']
    return reg['boards'][str(seed)]
W.check_seed=check
base=assets['parent'].read_text()
OLD=pack.literal(base,pack.STUDENT)
with np.load(assets['student_start'],allow_pickle=False) as original:
    assert pack.quantize(assets['student_start'],original)==OLD
    arrays={k:original[k] for k in original.files}
directions=[]
for module,k in [('whead',2),('bhead',1)]:
    weight=arrays[module+'.weight'];bias=arrays[module+'.bias']; m=np.column_stack([weight,bias]).astype(np.float64)
    u,s,vt=np.linalg.svd(m-m.mean(0,keepdims=True),full_matrices=False)
    for i in range(k):
        delta=s[i]*np.outer(u[:,i],vt[i])
        directions.append((module,delta))
KEYS={'whead.weight','whead.bias','bhead.weight','bhead.bias'}
def packaged(name,coeff,permanent=False):
    dst=(H if permanent else CACHE)/(name+'.py');npz=(H if permanent else CACHE)/(name+'.npz')
    sd={k:v.copy() if k in KEYS else v for k,v in arrays.items()}
    for value,(module,axis) in zip(coeff,directions):
        sd[module+'.weight']+=(value*axis[:,:-1]).astype(np.float32)
        sd[module+'.bias']+=(value*axis[:,-1]).astype(np.float32)
    if not any(coeff):
        if not permanent:return assets['parent'],{}
        npz.write_bytes(assets['student_start'].read_bytes());dst.write_bytes(assets['parent'].read_bytes())
        return dst,dict(pack_sha256=digest(dst),weights_sha256=digest(npz),q8_changed={},coefficient=list(coeff))
    tmp=npz.with_suffix('.part.npz')
    np.savez(tmp,**sd);tmp.replace(npz)
    with np.load(assets['student_start'],allow_pickle=False) as z:q=pack.quantize(npz,z)
    code=pack.substitute(base,q);assert pack.substitute(code,OLD)==pack.substitute(base,OLD)
    old_quant=pack.cnet2_pack.quantize(arrays)[2];new_quant=pack.cnet2_pack.quantize(sd)[2]
    changes={k:int((v!=old_quant[k]).sum()) for k,v in new_quant.items() if not np.array_equal(v,old_quant[k])}
    assert set(changes) and set(changes)<=KEYS
    dst.write_text(code)
    return dst,dict(pack_sha256=digest(dst),weights_sha256=digest(npz),q8_changed=changes,coefficient=list(coeff))
mods={}
def loaded(path):
    if path not in mods:mods[path]=imported('full_episode_public_'+str(len(mods)),path)
    return mods[path]
def play(arm,path,s,seat,fold):
    receipt=H/'games'/fold/f'{arm}_{s}_{seat}.json.gz'
    tid=dict(arm=arm,pack=digest(path),opponent=P['opponent']['sha256'],seed=s,seat=seat,terrain=check(s),protocol=reg['protocol'])
    if receipt.exists():
        d=read(receipt);assert d['identity']==tid;return d
    start=time.monotonic();world=W.World(s);assert tid['terrain']==world.terrain_sha256
    ours=loaded(path);other=loaded(assets['opponent']); ps=[ours,other] if seat==0 else [other,ours];tape=[];checkpoints={}
    while world.terminal is None:
        obs=[world.obs(i) for i in (0,1)]
        assert len(tape)==obs[0]['step']==obs[1]['step'] and obs[0]['player']==0 and obs[1]['player']==1
        if world.step in (0,60,100,150,200,250):
            checkpoints[str(world.step)]=[dict(cash=float(pl[0]),workers=len(pl[2]),trees=len(pl[3])) for pl in obs[0]['players']]
        commands=[ps[i].agent(obs[i],world.config) for i in (0,1)]
        world.advance(commands);tape.append(commands)
    result=W.replay_official(s,tape)
    assert result['terrain_sha256']==world.terrain_sha256 and all(result[k]==world.terminal[k] for k in ['steps','termination','extinct_by_seat','result_by_seat'])
    assert max(abs(v-w) for v,w in zip(result['scores'],world.terminal['scores']))<.005001
    d=dict(identity=tid,official=result,history=checkpoints,actions=tape,seconds=time.monotonic()-start)
    put(receipt,d)
    print(fold,arm,s,seat,result['steps'],round(result['u_by_seat'][seat],5),round(d['seconds'],1),flush=True)
    return d
def terminals(d):
    s=d['identity']['seat'];r=d['official'];return dict(u=r['u_by_seat'][s],margin=r['margin_by_seat'][s],steps=r['steps'],extinct=r['extinct_by_seat'][s],result=r['result_by_seat'][s],scores=r['scores'])
def pair_result(fold,listener):
    results=[]
    for seed in P[fold]:
        for seat in (0,1):
            a=play('parent',assets['parent'],seed,seat,fold);b=play('selected',listener,seed,seat,fold)
            x,y=terminals(a),terminals(b)
            results.append(dict(seed=seed,seat=seat,parent=x,selected=y,delta_U=y['u']-x['u'],delta_margin=y['margin']-x['margin']))
    by_map={s:float(np.mean([r['delta_U'] for r in results if r['seed']==s])) for s in P[fold]}
    by_seat={s:float(np.mean([r['delta_U'] for r in results if r['seat']==s])) for s in (0,1)}
    summarize=lambda a: dict(WDL=dict(Counter(r[a]['result'] for r in results)),mean_U=float(np.mean([r[a]['u'] for r in results])),mean_margin=float(np.mean([r[a]['margin'] for r in results])),extinctions=sum(r[a]['extinct'] for r in results),mean_steps=float(np.mean([r[a]['steps'] for r in results])))
    return dict(games=results,parent=summarize('parent'),selected=summarize('selected'),by_map=by_map,by_seat=by_seat,mean_delta_U=float(np.mean(list(by_map.values()))),mean_delta_margin=float(np.mean([r['delta_margin'] for r in results])))
def mechanism(pkg):
    m=loaded(pkg);times={0,20,40,60,80,100,125,150,175,200,225,250,275,297}
    groups={p:Counter() for p in (0,1,2)};witness=[]
    for seed in P['train']:
        for seat in (0,1):
            src=read(H/'games/train'/f'parent_{seed}_{seat}.json.gz');world=W.World(seed)
            for t,pair in enumerate(src['actions']):
                obs=world.obs(seat)
                if t in times:
                    issued=m.agent(obs,world.config); parent=pair[seat];phase=min(t//100,2);row=groups[phase];row['roots']+=1
                    row['changed']+=issued!=parent
                    if issued!=parent:witness.append(dict(seed=seed,seat=seat,t=t,parent=parent,selected=issued))
                else:m._FEAT.observe(obs)
                world.advance(pair)
    return dict(by_phase={k:dict(v) for k,v in groups.items()},changed_roots=witness,all_phases=all(v['changed']>0 for v in groups.values()))
def run():
    assert os.sched_getaffinity(0)=={3} and all(os.environ[k]=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
    sealed=H/'selection.json'
    if not sealed.exists():
        dtab=[];g=[]
        for axis in range(3):
            results={}; receipts={}
            for tag,sgn in [('plus',1.),('minus',-1.)]:
                k=f'axis{axis}_{tag}';vc=np.zeros(3);vc[axis]=sgn*.5;path,r=packaged(k,vc);receipts[tag]=r
                results[tag]={(s,seat):terminals(play(k,path,s,seat,'train')) for s in P['train'] for seat in (0,1)}
            rows=[dict(seed=s,seat=seat,plus=results['plus'][s,seat],minus=results['minus'][s,seat],delta_U=results['plus'][s,seat]['u']-results['minus'][s,seat]['u']) for s in P['train'] for seat in (0,1)]
            v=float(np.mean([r['delta_U'] for r in rows]));g.append(v);dtab.append(dict(axis=axis,identity=receipts,terminals=rows,directional_gradient=v))
        norm=float(np.linalg.norm(g));coef=.30*np.asarray(g)/norm if norm>1e-8 else np.zeros(3)
        fin,info=packaged('selected',coef,True)
        put(sealed,dict(protocol=reg['protocol'],source=digest(Path(__file__)),directions=dtab,gradient=g,coefficients=coef.tolist(),identity=info))
    fit=read(sealed); pk=H/'selected.py';assert digest(pk)==fit['identity']['pack_sha256'] and fit['source']==digest(Path(__file__))
    center=pair_result('train',pk);put(H/'training.json',center)
    bench=mechanism(pk);put(H/'mechanism.json',bench)
    held=pair_result('holdout',pk)
    checks={'positive_U':held['mean_delta_U']>0,'at_least_three_maps':sum(v>0 for v in held['by_map'].values())>=3,'both_seats_nonnegative':min(held['by_seat'].values())>=0,'extinctions_not_increased':held['selected']['extinctions']<=held['parent']['extinctions'],'issued_changes_in_all_three_phases':bench['all_phases']}
    put(H/'result.json',dict(selection=digest(sealed),holdout=held,checks=checks,proxy_gate=all(checks.values()),original_opponent_games=0))
    print('holdout',held['mean_delta_U'],held['mean_delta_margin'],checks,flush=True)
if __name__=='__main__':run()
