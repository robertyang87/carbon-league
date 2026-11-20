"""Reapply a genuine parent's executed schedule to both decoders, independently of timing noise."""
import copy,gzip,json,os,sys,types,zipfile
from pathlib import Path
from probe import HERE,PARENT,build,base,rr,fp,np
def module(data,label):
    text=data.decode().replace('\r\n','\n')
    for a,b in [('    cands = [act0]; keys =','    most = _replay_budget("menu",most)\n    cands = [act0]; keys ='),
                ('        return _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT','        return _replay_budget("cut", _time.perf_counter() - t0 + _S_SAFE * need > _S_ABORT)')]:
        assert text.count(a)==1;text=text.replace(a,b)
    mod=types.ModuleType(label);exec(compile(text,label,'exec'),mod.__dict__);return mod
def run():
    assert os.sched_getaffinity(0)==({3} if '--warm' in sys.argv else {4})
    os.environ['CARBON_DEBUG']='1'
    import torch
    torch.set_num_threads(1)
    plan=json.loads((HERE/'protocol.json').read_text())
    parent=module(PARENT.read_bytes(),'clock_parent');alt=module(build(),'clock_child')
    cfg=base.econ.config();cfg.pop('randomSeed',None)
    warmed="--warm" in sys.argv
    quota=Path('/sys/fs/cgroup/cpu.stat');begin=quota.read_text()
    rows=[];actor_checks=[]
    for ar in plan['archives']:
        with zipfile.ZipFile(HERE.parent/'cold_start/development/hosted'/(ar+'.zip')) as z:
            reg=json.loads(z.read('cold_repair/registration.json'))
            frames=gzip.decompress(z.read('cold_repair/observations.jsonl.gz')).splitlines()
            cmds=z.read('cold_repair/turns.jsonl').splitlines()
        s=reg['candidate_seat'];seed=reg['seed']
        tracker=parent.BeliefTracker(parent._Bel,s);prev=None
        rolling=[0.,0.,0.,0.]
        calibration=[]
        for t,raw in enumerate(frames):
            ob=json.loads(raw)['observations'][s];public=ob['players'];q=json.loads(cmds[t]);cmd=q['calls'][s]['response']['action']
            assert base.digest(ob)==q['public_observation_sha256'][s]
            seed_t=seed+t*193+s
            before=list(rolling)
            if warmed and 268<=t<=max(plan['steps']):
                parent._replay_budget=lambda k,v:v
                receipt=rr.call(parent,(tracker,prev),ob,cfg,seed_t,rolling,'P')
                rolling=list(receipt['clocks']['cost']);calibration.append(dict(t=t,clock=receipt['clocks']))
            costs=before if warmed else plan['costs']
            if t in plan['steps']:
                gates={'cut':[],'menu':[]};calls=[]
                for m,mode in [(parent,'capture'),(parent,'parent'),(alt,'restored')]:
                    idx=dict(cut=0,menu=0)
                    def gate(k,v):
                        if mode=='capture':gates[k].append(v)
                        i=idx[k];idx[k]+=1
                        return gates[k][i]
                    m._replay_budget=gate
                    r=rr.call(m,(tracker,prev),ob,cfg,seed_t,costs,'P')
                    calls.append({k:r[k] for k in ['menu','depths','values','action','selected','clocks']})
                    assert all(len(gates[k])==idx[k] for k in idx)
                c,p,a=calls
                for k in ['action','menu','depths','values','selected']: assert c[k]==p[k]
                assert p['menu']==a['menu'] and all(p['clocks'][k]==a['clocks'][k] for k in ('K','H','rolls'))
                native=[]
                if warmed:
                    for mode in ('PAAP' if len(rows)%2==0 else 'APPA'):
                        m=parent if mode=='P' else alt;m._replay_budget=lambda k,v:v
                        event=rr.call(m,(tracker,prev),ob,cfg,seed_t,costs,'P')
                        native.append(dict(mode=mode,**{k:event[k] for k in ('clocks','action','menu','selected','depths','values')}))
                rows.append(dict(seed=seed,seat=s,t=t,costs=costs,prior_local_clocks=calibration[:-1] if warmed else [],natural_abba=native,reference=c,corrected=a,budget=gates,
                    changed=p['action']!=a['action'],same_each_birth_depth=p['depths']==a['depths'],values_changed=p['values']!=a['values']))
            if t>=284:
                # Independent public actor contract. Both decode exactly the same policy logits and sample stream.
                t2=copy.deepcopy(tracker);est,known=t2.step(public,t,ob['carbon'])
                x=parent._quant(parent.encode(public,t,s,est,known,prev))
                lw,lb=parent._POLICY.forward(x,parent._SID);key=seed+t*193+s
                parent._ST=dict(tr=copy.deepcopy(tracker),prev=prev,rng=np.random.RandomState(key))
                issued=parent.agent(ob,None)
                direct=alt._sample(public,s,lw,lb,np.random.RandomState(key),t,home_x=x)
                stock=parent._sample(public,s,lw,lb,np.random.RandomState(key),t)
                assert direct==issued
                delta=[]
                for uid,u in public[s][2].items():
                    o=stock.get(uid,'NONE');a=issued.get(uid,'NONE')
                    if o!=a:delta.append(dict(uid=uid,position=u[0],cargo=u[1],role=u[2],stock=o,aligned=a,real=cmd.get(uid,'NONE')))
                actor_checks.append(dict(seed=seed,seat=s,t=t,changed=stock!=direct,unit_changes=delta))
            tracker.step(public,t,ob['carbon']);tracker.set_actions(public,cmd);prev=public
    rr.save(HERE/('warmed.json.gz' if warmed else 'controlled.json.gz'),dict(protocol_sha256=fp((HERE/'protocol.json').read_bytes()),rows=rows,public_actor_checks=actor_checks,cgroup_cpu_stat_before=begin,cgroup_cpu_stat_after=quota.read_text(),affinity=sorted(os.sched_getaffinity(0))))
    print('fixed-work',[(r['seed'],r['seat'],r['t']) for r in rows if r['changed']],'anyValue',sum(r['values_changed'] for r in rows),flush=True)
    print('actor',len(actor_checks),sum(r['changed'] for r in actor_checks),flush=True)
if __name__=='__main__':run()
