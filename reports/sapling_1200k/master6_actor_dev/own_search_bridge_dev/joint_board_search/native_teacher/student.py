from labels import *
from audit import module
def run():
    p=read(HERE/'student_protocol.json');rows=[];skips=[];cost=0
    for c in p['cards']:
        d=read(HERE/f"labels_{c['seed']}_{c['seat']}.json.gz");assert d['protocol']==sha(HERE/'student_protocol.json');cost+=d['queries']
        for r in d['root_data']:
            if not r['complete']:
                skips.append([c['seed'],c['seat'],r['t']]);continue
            y=np.array([0]+[np.mean(pair['final']['values'][1])-np.mean(pair['final']['values'][0]) for pair in r['pairs']]);rows.append((c,r,np.array(r['x']),y))
    assert cost<=90
    fit=[(x[1:],y[1:]) for c,r,x,y in rows if c['split']=='fit'];assert fit
    rms=np.sqrt(np.mean([np.mean(x*x,axis=0) for x,y in fit],axis=0)).clip(.001)
    Z=np.concatenate([x/rms/np.sqrt(len(x)) for x,y in fit]);Y=np.concatenate([y/np.sqrt(len(x)) for x,y in fit])
    coef=(np.linalg.solve(Z.T@Z+len(fit)*np.eye(18),Z.T@Y)/rms).astype('f')
    f=HERE/'head.npz';np.savez(f,coefficient=coef)
    src=(J/'candidate.py').read_bytes();tail=((HERE/'head_runtime.py').read_text()+'\n_HEAD_COEF=np.array('+repr(coef.tolist())+',dtype=np.float32)\n').encode()
    candidate=HERE/'candidate.py';candidate.write_bytes(src+b'\n'+tail);assert len(candidate.read_bytes())<10000000
    labels=[];free={};cfg=read(ROOT/source_cards()[0]['source'])['game_configuration']
    for c,r,x,y in rows:
        guess=x@coef;i=int(np.argmax(guess));labels.append(dict(seed=c['seed'],seat=c['seat'],split=c['split'],t=r['t'],root_menu_count=len(y),ref_choice=int(np.argmax(y)),pick=i,
            saved_teacher_gain=float(y[i]),max_teacher_gain=float(y.max()),rmse=float(np.mean((guess[1:]-y[1:])**2)),zero_mse=float(np.mean(y[1:]**2))))
        free[(c['seed'],c['seat'],r['t'])]=r['commands'][i]
    entries=[]
    for c in p['cards']:
        if c['split']!='held':continue
        g=read(ROOT/c['source']);s=c['seat'];sim=bridge.FastCarbon(cfg).reset(seed=c['seed']);m=module(candidate,'held_search_student')
        for t,a in enumerate(g['actions'][:225]):
            obs=sim.obs(s)
            if t==0 or t in p['steps']:
                w=time.perf_counter();actual=m.agent(obs,cfg);wall=time.perf_counter()-w;idx=(c['seed'],s,t)
                if idx in free:
                    assert actual==free[idx] and m._JB_INFO['base']==a[s]
                    entries.append(dict(seed=c['seed'],seat=s,t=t,changed=actual!=a[s],wall=wall,status=m._JB_INFO['status']))
            else:m._FEAT.observe(obs);m._JB_FEAT.observe(m.public_enemy(obs,m._FEAT.est))
            sim.step(*a)
    scored=[]
    cards=source_cards();by=defaultdict(list)
    for c in cards:by[c['source']].append(c)
    m=module(candidate,'original16_student')
    for src,cc in by.items():
        g=read(ROOT/src);sim=bridge.FastCarbon(g['game_configuration']).reset(seed=g['seed']);s=cc[0]['seat'];views=[m.F.Feat(j) for j in (0,1)];needed={c['t']:c for c in cc}
        for t in range(max(needed)+1):
            obs=sim.obs(s);views[s].observe(obs);ox=views[1-s].observe(m.public_enemy(obs,views[s].est))
            if t in needed:
                c=needed[t];i,rr,_=m._joint_search(sim.config,obs,views,ox,c['cmds'],time.perf_counter()+3);assert rr
                delta=c['margins'][1]-c['margins'][0];scored.append(dict(case=c['case'],t=t,seed=c['seed'],seat=s,pick=i,score=rr[0]['arms'],correct=i==int(delta>0),selected_gain=delta*i))
            sim.step(*g['actions'][t])
    detail={}
    for split in ('fit','held'):
        ss=[r for r in labels if r['split']==split];detail[split]={}
        for seed in sorted({r['seed'] for r in ss}):
            v=[r for r in ss if r['seed']==seed];detail[split][seed]=dict(roots=len(v),matches=sum(z['ref_choice']==z['pick'] for z in v),changed=sum(z['pick']!=0 for z in v),
                gain=float(np.mean([z['saved_teacher_gain'] for z in v])),teacher_best=float(np.mean([z['max_teacher_gain'] for z in v])),
                mse=float(np.mean([z['rmse'] for z in v])),zero_mse=float(np.mean([z['zero_mse'] for z in v])))
    n=sum(x['correct'] for x in scored);gain=float(np.mean([x['selected_gain'] for x in scored]));held=list(detail['held'].values())
    gate=(len(held)==2 and all(z['gain']>0 for z in held) and np.mean([z['mse'] for z in held])<np.mean([z['zero_mse'] for z in held]) and n>8 and gain>0)
    bridge.save(HERE/'student_results.json',dict(protocol=sha(HERE/'student_protocol.json'),source=sha(__file__),head=sha(f),candidate_sha256=sha(candidate),
                candidate_bytes=candidate.stat().st_size,self_search_parent=sha(J/'candidate.py'),new_parameters=len(coef),teacher_queries=cost,roots=len(rows),skipped=skips,
                records=labels,summary=detail,held_free=entries,previous_original16=scored,native_correct=n,native_gain=gain,send_to_match=bool(gate)))
if __name__=='__main__':run()
