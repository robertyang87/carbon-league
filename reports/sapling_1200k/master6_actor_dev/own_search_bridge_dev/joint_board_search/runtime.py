"""Shared NumPy board encoding and small finite complete-round search. Appended to the frozen parent."""
import time,copy
def _small_linear(a,name):
    return a@_SW[name+'.weight'].T+_SW[name+'.bias']
def _board_batch(xs):
    z=np.stack([x['maps'].T for x in xs]).astype(np.float32)/255
    h=np.tanh(_small_linear(np.concatenate([z,z[:,F.DEST[:,1:]].mean(2)],-1),'cell'))
    c=np.tanh(_small_linear(np.concatenate([h.mean(1),h.max(1),np.stack([x['scal'] for x in xs])],-1),'context'))
    return h,c,_small_linear(c,'value')
def _small_orders(xs,feats,board,context):
    B=len(xs);ans=np.full((B,12),-1,int);pf=[F.Prefix(x,f.aux) for x,f in zip(xs,feats)]
    for j in range(12):
        live=[i for i,x in enumerate(xs) if x['tok_mask'][j]]
        if not live:continue
        inp=[];oks=[]
        for i in live:
            x=xs[i];pos=int(x['tok_pos'][j]);u,m,legal=pf[i].feats(j)
            cells=F.DEST[pos] if j<10 else np.full(5,pos);role=min(int(x['tok_kind'][j]),2)
            inp.append(np.concatenate((board[i,cells],np.broadcast_to(context[i],(5,16)),np.broadcast_to(np.eye(3,dtype='f')[role],(5,3)),np.eye(5,dtype='f'),np.broadcast_to(u,(5,len(u))),m),axis=-1));oks.append(legal)
        logits=_small_linear(np.tanh(_small_linear(np.stack(inp),'pol')),'last')[...,0]
        for i,a in zip(live,np.argmax(np.where(oks,logits,-np.inf),axis=1)):
            ans[i,j]=a;pf[i].advance(j,int(a))
    return [f.to_actions(y) for f,y in zip(feats,ans)]
def _score(cash,step_value,terminal=False):
    if terminal:
        m=cash;u=.5*(1. if m>0 else .5 if m==0 else 0.)+.5/(1+np.exp(-np.clip(m/1000,-40,40)));pred=m
    else:
        u=1/(1+np.exp(-np.clip(step_value[0],-40,40)));pred=cash+4000*step_value[1]
    return .5*(2*u-1)+.5*np.tanh(pred/2000)
def _guard_pair(x,base,res):
    protected=(np.arange(12)>=10)|(x['tok_kind'][:12]==1)
    return base if np.any(base[protected]!=res[protected]) or _return_conflicts(x,res)-_return_conflicts(x,base) else res
def _root_options(x,aux,enc,original,f):
    caps={};old=_jb_decode(x,aux,enc,False,{},caps)
    base=f.to_actions(original);coll=_return_conflicts(x,original)
    rows=[]
    for j,(lg,ok) in caps.items():
        if j>=10:continue
        ix=np.flatnonzero(ok);order=ix[np.argsort(-lg[ix],kind='stable')]
        if len(order)>1:rows.append((float(lg[order[0]]-lg[order[1]]),j))
    candidates=[base];details=[{'kind':'parent','forced':[]}]
    if len(rows)<2:return candidates,details
    def add(slots,kind):
        forced={j:1 for j in slots}
        two=[_jb_decode(x,aux,enc,mode,forced,{}) for mode in (False,True)]
        new=_guard_pair(x,*two);cmd=f.to_actions(new)
        if cmd in candidates or _return_conflicts(x,new)-coll:return
        changed=np.flatnonzero(new!=original)
        if len(slots)>1 and (len(changed)<2 or not set(slots)<=set(changed.tolist())):return
        candidates.append(cmd);details.append(dict(kind=kind,forced=slots,changed=changed.tolist()))
    rows.sort()
    primary=rows[0][1];add([primary],'single')
    pairs=sorted((int(F.DIST[x['tok_pos'][a],x['tok_pos'][b]]),ga+gb,a,b) for ia,(ga,a) in enumerate(rows) for gb,b in rows[ia+1:])
    for _,__,a,b in pairs[:6]:
        add([a,b],'paired')
        if len(candidates)>=4:break
    return candidates,details
def _joint_search(cfg,obs,views,opp_x,commands,deadline):
    s=obs['player'];op=1-s
    b,c,v=_board_batch([opp_x]);enemy=_small_orders([opp_x],[views[op]],b,c)[0]
    state=estimate_view(cfg,obs,views[s]).get_state();arms=[]
    for cmd in commands:
        sim=FastCarbon(cfg);sim.set_state(state);pair=[cmd,enemy] if s==0 else [enemy,cmd]
        arms.append(dict(sim=sim,views=copy.deepcopy(views),actions=pair,done=False))
    ids=list(range(len(arms)));best=0;rounds=[]
    for d in range(1,6):
        if time.perf_counter()>=deadline:return best,rounds,enemy
        batch=[];fs=[];advance=[]
        for i in ids:
            a=arms[i];sim=a['sim']
            if a['done']:continue
            pre=list(sim.cash);sim.step(*a['actions'])
            margin,done=leaf(sim,s,pre)
            if done:
                a.update(done=True,score=float(_score(margin,None,True)));continue
            advance.append(i)
            for seat in (0,1):
                f=a['views'][seat];batch.append(f.observe(sim.obs(seat)));fs.append(f)
        if batch:
            b,c,v=_board_batch(batch)
            orders=_small_orders(batch,fs,b,c) if d<5 else []
            for j,i in enumerate(advance):
                a=arms[i];cash=a['sim'].cash
                focal=float(_score(cash[s]-cash[op],v[2*j+s]))
                rival=float(_score(cash[op]-cash[s],v[2*j+op]))
                a['score']=(focal-rival)/2
                if d<5:a['actions']=orders[2*j:2*j+2]
        if time.perf_counter()>=deadline:return best,rounds,enemy
        if d in (1,3,5):
            ordered=sorted(ids,key=lambda i:(-arms[i]['score'],i));best=ordered[0];rounds.append(dict(depth=d,arms={i:float(arms[i]['score']) for i in ids},best=best))
            if d<5:ids=[0]+[i for i in ordered if i!=0][:2 if d==1 else 1]
    return best,rounds,enemy
_JB_ORIGINAL=_mixed_actions
_JB_FEAT=None
_JB_PUBLIC=None
_JB_INFO={}
def _mixed_actions(x,aux):
    global _JB_INFO,_JB_FEAT
    obs,cfg,start=_JB_PUBLIC;s=obs['player'];enc=[]
    for pol in _NETS:
        t,c=pol.encode(x,_HYBRID['style']);enc.append((pol,t,c,pol.lin(c,'h0')))
    ref=_return_decode(x,aux,enc,False);res=_return_decode(x,aux,enc,True);orig=_guard_pair(x,ref,res)
    _JB_INFO=dict(status='fallback',base=_FEAT.to_actions(orig),rounds=[])
    try:
        if _JB_FEAT is None or obs['step']==0 or _JB_FEAT.seat!=1-s:_JB_FEAT=F.Feat(1-s)
        opp=_JB_FEAT.observe(public_enemy(obs,_FEAT.est))
        if time.perf_counter()-start>.42:return orig
        cmds,kinds=_root_options(x,aux,enc,orig,_FEAT)
        if len(cmds)<2:return orig
        views=[_FEAT,_JB_FEAT] if s==0 else [_JB_FEAT,_FEAT]
        win,rounds,e=_joint_search(cfg,obs,views,opp,cmds,start+.68)
        okay=time.perf_counter()-start<.80 and bool(rounds)
        _JB_INFO.update(status='complete' if okay else 'timed_out',public_enemy=e,rounds=rounds,candidates=cmds,kinds=kinds,chosen=win if okay else 0)
        return _FEAT.labels(cmds[win]) if okay else orig
    except (ArithmeticError,ValueError,KeyError,IndexError,RuntimeError) as err:
        _JB_INFO.update(status='error',error=repr(err));return orig
_JB_PARENT_AGENT=agent
def agent(observation,configuration=None):
    global _JB_PUBLIC
    start=time.perf_counter()
    obs={k:observation[k] for k in ('player','step','players','trees','carbon','remainingOverageTime') if k in observation}
    _JB_PUBLIC=(obs,{**_JB_CFG,**(configuration or {})},start)
    return _JB_PARENT_AGENT(obs,configuration)
