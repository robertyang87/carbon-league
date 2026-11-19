"""Only self-owned encoders and simulated opponent actions are available in this deployed scorer."""
def _head_rows(cfg,obs,views,opp_x,commands):
    s=obs['player'];o=1-s
    b,c,_=_board_batch([opp_x]);enemy=_small_orders([opp_x],[views[o]],b,c)[0]
    initial=estimate_view(cfg,obs,views[s]).get_state();xs=[];cash=[];terminal=[]
    for move in commands:
        sim=FastCarbon(cfg);sim.set_state(initial);before=list(sim.cash)
        sim.step(*([move,enemy] if s==0 else [enemy,move]))
        mar,done=leaf(sim,s,before);cash.append(sim.cash[s]-sim.cash[o]);terminal.append(float(_score(mar,None,True)) if done else None)
        for seat in [s,o]:
            f=copy.deepcopy(views[seat]);xs.append(f.observe(sim.obs(seat)))
    _,ctx,v=_board_batch(xs);a=ctx[::2];b=ctx[1::2];cash=np.asarray(cash)/1000
    val=np.array([(_score(cash[i]*1000,v[2*i])-_score(-cash[i]*1000,v[2*i+1]))/2 for i in range(len(commands))])
    out=np.column_stack([a-b,cash,val]);return out-out[:1],enemy,terminal
def _joint_search(cfg,obs,views,opp_x,commands,deadline):
    x,enemy,terminal=_head_rows(cfg,obs,views,opp_x,commands)
    if time.perf_counter()>=deadline:return 0,[],enemy
    scores=x@_HEAD_COEF
    # Exact common-step outcomes, if EVERY candidate terminates immediately.
    if all(v is not None for v in terminal):scores=np.array(terminal)
    pick=int(np.argmax(scores));return pick,[dict(depth=1,arms={i:float(v) for i,v in enumerate(scores)},best=pick)],enemy
