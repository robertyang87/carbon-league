"""Build symmetric perturbations around the sealed it50/Sapling center and verify public command coverage."""
import copy,json,os,shutil,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from register_fit import ROOT,HERE,FULL,ACTOR,sha
sys.path.insert(0,str(FULL));import full as f
P=f.read(HERE/'protocol.json'); work=Path('/workspace/sapling-direct-original-signs-09b4');work.mkdir(exist_ok=True)
center_path=ROOT/P['center']['path'];fp=ROOT/P['center_npz']['path'];base=center_path.read_text()
with np.load(fp,allow_pickle=False) as w:
    center={k:w[k] for k in w.files}
    assert f.pack.quantize(fp,w)==f.pack.literal(base,f.pack.STUDENT)
cs=np.asarray(f.read(FULL/'selection.json')['coefficients'])
d0,d1=f.directions[0][1],f.directions[1][1]
axis=np.array([-cs[1]*(d1*d1).sum(),cs[0]*(d0*d0).sum()]);axis/=np.linalg.norm(axis)
D=axis[0]*d0+axis[1]*d1
def dotcenter():
    a=np.column_stack([center['whead.weight']-f.arrays['whead.weight'],center['whead.bias']-f.arrays['whead.bias']])
    return float((D*a).sum()/np.linalg.norm(a)/np.linalg.norm(D))
def make(radius,label,folder):
    out=folder/(label+'.npz');py=folder/(label+'.py')
    if radius==0:
        shutil.copyfile(fp,out);shutil.copyfile(center_path,py)
        return dict(pack=sha(py),npz=sha(out),changed_quantized={},radius=0)
    w=dict(center);w['whead.weight']=center['whead.weight']+(radius*D[:,:-1]).astype(np.float32)
    w['whead.bias']=center['whead.bias']+(radius*D[:,-1]).astype(np.float32)
    np.savez(out,**w)
    with np.load(fp,allow_pickle=False) as z:q=f.pack.quantize(out,z)
    prev=f.pack.literal(base,f.pack.STUDENT);code=f.pack.substitute(base,q)
    assert f.pack.substitute(code,prev)==f.pack.substitute(base,prev)
    py.write_text(code)
    old=f.pack.cnet2_pack.quantize(center)[2];new=f.pack.cnet2_pack.quantize(w)[2]
    delta={k:int((v!=old[k]).sum()) for k,v in new.items() if not np.array_equal(v,old[k])}
    assert set(delta)=={'whead.weight','whead.bias'}
    return dict(pack=sha(py),npz=sha(out),changed_quantized=delta,radius=radius)
def audit_maps():
    r=f.read(FULL/'register.json');p0=f.read(FULL/'direct_original/clearance.json')['physical_boards']
    exclude=set(r['boards'].values())|set(p0.values())
    orig=f.read(ACTOR/'actor_partner_dev/protocol.json');assert sorted(orig['maps'])==P['seeds']
    boards={}
    for ref in P['historic_native_tapes']:
        path=ROOT/ref['path'];assert sha(path)==ref['sha256']
        d=f.read(path);s=d['seed'];k=str(s);t=d['initial_terrain_u8_sha256'];assert not d['errors'] and s in P['seeds'] and t not in exclude
        assert d['examiner_md5']=='57f9a594cce4f2ec9e3573d8d9d26ca1' and (k not in boards or boards[k]==t)
        boards[k]=t
    assert len(boards)==len(set(boards.values()))==2 and set(P['seeds']).isdisjoint(map(int,r['boards'])) and '621758343' not in boards
    f.put(HERE/'maps.json',dict(protocol=sha(HERE/'protocol.json'),boards=boards,prior_development=P['previous_development'],recent_maps_blocked=sorted(exclude),use='Previously exposed real original-examiner development boards now used for optimization. Exclude from future assessment.'))
def calibration(radius):
    dst=work/f'r{int(radius*100):03}';dst.mkdir(exist_ok=True)
    recipes={k:make(radius*v,k,dst) for k,v in [('plus',1),('minus',-1)]}
    paths={k:dst/(k+'.py') for k in recipes};paths['center']=center_path
    models={k:f.loaded(path) for k,path in paths.items()}
    counts=Counter();phases=Counter();seats=Counter();examples=[];N=0
    for ref in P['calibration_center_games']:
        path=ROOT/ref['path'];assert sha(path)==ref['sha256'];d=f.read(path);side=d['candidate_seat']
        assert d['pack_sha256']==P['center']['sha256'] and not d['errors']
        seed=d['seed'];f.W.check_seed=lambda s: d['initial_terrain_u8_sha256'] if s==seed else (_ for _ in ()).throw(ValueError('wrong source'))
        world=f.W.World(seed);assert world.terrain_sha256==d['initial_terrain_u8_sha256']
        for t,pair in enumerate(d['actions']):
            obs=world.obs(side)
            if t%3==0 and t<=297:
                actions={k:m.agent(obs,world.config) for k,m in models.items()}
                assert actions['center']==pair[side]
                N+=1
                for k in ('plus','minus'):counts[k]+=actions[k]!=actions['center']
                if actions['plus']!=actions['minus']:
                    phases[min(2,t//100)]+=1;seats[side]+=1
                    examples.append(dict(seat=side,t=t,actions=actions))
            else:
                for m in models.values():m._FEAT.observe(obs)
            world.advance(pair)
    ok=(all(3<=counts[k]<=.30*N for k in recipes) and all(phases[k]>0 for k in range(3)) and all(seats[k]>0 for k in (0,1)))
    return dst,recipes,dict(radius=radius,n=N,changes_vs_center=dict(counts),plus_minus_changes_by_phase=dict(phases),plus_minus_changes_by_seat=dict(seats),changed_public_commands=examples,eligible=ok)
def main():
    audit_maps();assert abs(dotcenter())<1.e-6
    trials=[]
    for h in P['radii']:
        dst,rec,z=calibration(h);trials.append(z)
        print(h,z['changes_vs_center'],z['plus_minus_changes_by_phase'],z['eligible'],flush=True)
        if z['eligible']:
            for k in rec:
                for suf in ('.py','.npz'):shutil.copyfile(dst/(k+suf),HERE/(k+suf))
                assert sha(HERE/(k+'.py'))==rec[k]['pack'] and sha(HERE/(k+'.npz'))==rec[k]['npz']
            break
    else:rec={}
    f.put(HERE/'binding.json',dict(protocol=sha(HERE/'protocol.json'),source=sha(HERE/'prepare.py'),map_audit=sha(HERE/'maps.json'),axis=axis.tolist(),cosine_with_previous_update=dotcenter(),trials=trials,signs=rec))
if __name__=='__main__':main()
