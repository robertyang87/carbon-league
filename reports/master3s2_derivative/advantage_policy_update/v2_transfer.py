"""Verify the two sealed parent actors and run their transported ordinary BC policy under native search."""
import ast
import json
import hashlib
import numpy as np
from probe import replay_one
from data import P,HERE,D,module,read,sha,save
from build import assemble,V2_SHA

def main():
    src=assemble('zero');live=assemble('zero',True);child=assemble('bc',True)
    frozen=json.loads((HERE/'v2_transfer_protocol.json').read_bytes())
    assert sha(live)==frozen['source_online']==V2_SHA and sha(child)==frozen['candidate']
    def nodes(raw):
        return {n.name:ast.dump(n) for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    first,second=nodes(src),nodes(live)
    methods=('_Policy','_act','encode','_sample','go_home','BeliefTracker','_WinoNet','_scalars')
    assert all(first[n]==second[n] for n in methods)
    ref=module(src.replace(b'        return lw.T, lb.T',b'        return hf.T, lw.T, lb.T'),'reference_for_transfer')
    models={'v2':module(live,'online'), 'bc':module(child,'online_bc')}
    parent=models['v2'];candidate=models['bc']
    assert set(ref._POLICY.p)==set(parent._POLICY.p)
    params={k:sha(np.ascontiguousarray(v).tobytes()) for k,v in ref._POLICY.p.items()}
    assert all(np.array_equal(v,parent._POLICY.p[k]) and np.array_equal(v,candidate._POLICY.p[k]) for k,v in ref._POLICY.p.items())
    assert ref._SID==parent._SID==candidate._SID
    schema=json.loads((D.parents[1]/'code/env_src/carbon_challenge-master/zerosum_env/envs/carbon/carbon.json').read_bytes())
    cfg={k:v['default'] for k,v in schema['configuration'].items() if isinstance(v,dict) and 'default' in v}
    with np.load(HERE/'weights.npz') as f:book={k:f[k] for k in f.files}
    init=list(parent._S_COST);rows=[];errors={'default':0.,'worker':0.,'base':0.}
    for card in [c for c in P['tickets'] if c['split']=='held']:
        tape,_=read(card);s=card['seat']
        tr=parent.BeliefTracker(parent._Bel,s);tr2=ref.BeliefTracker(ref._Bel,s);prev=None
        for t,(obs,active) in enumerate(tape):
            if t in (46,165):
                key=int.from_bytes(hashlib.sha256(f"awr-native/{card['archive_sha256']}/{s}/{t}".encode()).digest()[:4],'big')
                row=dict(seed=card['seed'],map=card['map'],seat=s,t=t,runs=[])
                order=['v2','bc','bc','v2'] if len(rows)%2==0 else ['bc','v2','v2','bc']
                for arm in order:
                    row['runs'].append(replay_one(models[arm],arm,obs,cfg,tr,prev,key,init))
                v=[r for r in row['runs'] if r['mode']=='v2'];a=[r for r in row['runs'] if r['mode']=='bc'];r=v[0]
                row['repeated']=all(u[0]['issued']==u[1]['issued'] and u[0]['menu']==u[1]['menu'] for u in (v,a))
                row['identical_work_by_index']=all(z['depths']==r['depths'] and z['K']==r['K'] and z['M']==r['M'] and z['H']==r['H'] for z in row['runs'])
                row['identical_work_total']=all(sorted(z['depths'])==sorted(r['depths']) and z['K']==r['K'] and z['M']==r['M'] and z['H']==r['H'] for z in row['runs'])
                row['enemy_fixed']=all(z['enemy']==r['enemy'] for z in row['runs'])
                row['menu_changed']=r['menu']!=a[0]['menu'];row['first_changed']=r['first_menu']!=a[0]['first_menu']
                row['issued_changed']=r['issued']!=a[0]['issued'];row['changed_units']=[
                    dict(id=k,baseline=r['issued'].get(k,'NONE'),bc=a[0]['issued'].get(k,'NONE'),
                         historical=active[s].get(k,'NONE')) for k in sorted(set(r['issued'])|set(a[0]['issued'])) if r['issued'].get(k)!=a[0]['issued'].get(k)]
                row['attributed']=row['issued_changed'] and row['repeated'] and row['identical_work_by_index'] and row['enemy_fixed']
                row['attributed_counted_work']=row['issued_changed'] and row['repeated'] and row['identical_work_total'] and row['enemy_fixed']
                rows.append(row)
            (est,known),(e2,k2)=(r.step(obs['players'],t,obs['carbon']) for r in (tr,tr2))
            assert np.array_equal(est,e2) and np.array_equal(known,k2)
            if t in (46,165):
                x=parent._quant(parent.encode(obs['players'],t,s,est,known,prev)).astype(np.float32)
                x2=ref._quant(ref.encode(obs['players'],t,s,e2,k2,prev)).astype(np.float32); assert np.array_equal(x,x2)
                hf,w,b=ref._POLICY.forward(x,ref._SID)
                a0=parent._POLICY.forward(x,parent._SID)
                neutral=candidate._POLICY.forward(x,candidate._SID)
                adjusted=candidate._POLICY.forward(x,candidate._SID,own_root=True)
                for idx,(role,z) in enumerate((('worker',w),('base',b))):
                    assert np.array_equal(a0[idx],z) and np.array_equal(neutral[idx],z)
                    mu,sd,B=[book['bc_'+role+'_'+k] for k in ('mu','sd','B')]
                    expected=z+1.25*np.tanh((((hf-mu)/sd)@B[:-1]+B[-1])/1.25)
                    errors[role]=max(errors[role],float(abs(expected-adjusted[idx]).max()))
            for tracker in (tr,tr2):tracker.set_actions(obs['players'],active[s])
            prev=obs['players']
    assert max(errors.values())<.0002
    keys=('issued_changed','menu_changed','first_changed','repeated','identical_work_by_index','identical_work_total','enemy_fixed','attributed','attributed_counted_work')
    counts={k:sum(r[k] for r in rows) for k in keys}
    save(HERE/'v2_native.json.gz',dict(source=P['parent'],online=V2_SHA,candidate=sha(child),actor_parameter_hashes=params,sid=parent._SID,A0_funcs_equal=list(methods),output_error=errors,rows=rows,counts=counts))
    print(counts,errors,flush=True)

if __name__=='__main__':main()
