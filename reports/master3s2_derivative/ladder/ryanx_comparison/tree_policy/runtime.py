def _plant_history(tracker, obs):
    step = obs['step']
    old = getattr(tracker, '_plant_public_history', {})
    if step == 0 or any(t >= step for t in old):
        old = {}
    history = {lag: old[step-lag] for lag in (3,5) if step-lag in old}
    # The two lagged features read only the former position of the same enemy planter.
    enemy = 1 - obs['player']
    saved = [(None,None,{}), (None,None,{})]
    saved[enemy][2].update({uid:(w[0],) for uid,w in obs['players'][enemy][2].items() if w[2]=='PLANTER'})
    tracker._plant_public_history = {t:v for t,v in old.items() if t>step-5}
    tracker._plant_public_history[step] = saved
    return history


def _plant_reprice(obs, cfg, tracker, logits):
    history = _plant_history(tracker,obs)
    if cfg is None or obs['step']>=cfg['episodeSteps']-1:
        return
    ps=obs['players']; me=obs['player']; own=ps[me]
    price=_PLANT_LIB['quoted_price'](ps,me,cfg)
    if price<=0 or own[0]<price:
        return
    blocked={c for p in ps for c in p[1].values()} | {v[0] for p in ps for v in p[3].values()}
    for uid,(cell,_,role) in own[2].items():
        if role!='PLANTER' or cell in blocked:
            continue
        x=_PLANT_LIB['features'](obs,uid,cfg,history)
        net=_PLANT_MEAN+sum((v-m)/s*w for v,m,s,w in zip(x,_PLANT_XMEAN,_PLANT_SCALE,_PLANT_COEF))
        if net<0:
            logits[cell,0]+=max(-1.,net/price)
