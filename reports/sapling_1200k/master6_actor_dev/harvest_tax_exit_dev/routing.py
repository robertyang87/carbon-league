"""End a taxed harvest when preserving freight exceeds the foregone extra harvest."""
def choose(obs,commands,cfg,dist,dest,directions, counts=None):
    def count(key):
        if counts is not None: counts[key]+=1
    seat=obs['player'];own,foe=obs['players'][seat],obs['players'][1-seat]
    home=list(own[1].values());trees={p for who in obs['players'] for p,age in who[3].values()}
    hostile=[p for p,age in foe[3].values() if age<cfg['treeLifespan']]
    if not home or not hostile:return None
    def charge(cell):
        return min(1.,cfg['collectorAbsorptionRate']*sum(
          min(abs(cell%15-p%15),15-abs(cell%15-p%15))<=1 and
          min(abs(cell//15-p//15),15-abs(cell//15-p//15))<=1 for p in hostile))
    rate=max(0.,cfg['initialCollectRate']-sum(w[2]=='COLLECTOR' for w in own[2].values())*cfg['collectDecreaseRate'])
    lands={k:int(dest[w[0],directions.index(commands.get(k))]) for k,w in own[2].items()}
    recruit={pos for k,pos in own[1].items() if commands.get(k)}
    opts=[]
    for uid,(p,cargo,role) in own[2].items():
        if role!='COLLECTOR' or commands.get(uid) or cargo<=0 or p in home:continue
        current=charge(p)
        if not current:continue
        count("stays in enemy tree radius with cargo")
        carbon=obs['carbon'][p]
        if carbon is None or carbon<0:continue
        count("own cell carbon present")
        gross=0. if p in trees else rate*carbon
        stay_toll=current*(cargo+gross)
        initial=min(dist[p,b] for b in home)
        occupied={cell for k,cell in lands.items() if k!=uid}|recruit
        for i,z in enumerate(dest[p]):
            z=int(z)
            if z in occupied or z in foe[1].values() or min(dist[z,b] for b in home)!=initial-1:continue
            count("available homeward cell")
            if z not in home and any(k=='COLLECTOR' and c<=cargo and dist[z,ep]<=1 for ep,c,k in foe[2].values()):continue
            count("safe homeward cell")
            new_toll=0. if z in home else charge(z)*cargo
            advantage=2*(stay_toll-new_toll)-gross
            if stay_toll>new_toll:count("cheaper exposure")
            if advantage>0:count("positive joint asset estimate")
            if advantage>8:opts.append((-advantage,uid,i,cargo,gross,stay_toll,new_toll,p,z))
    if not opts:return None
    n,uid,i,cargo,gross,was,now,p,z=min(opts)
    return dict(worker=uid,old='NONE',alternative=directions[i],load=cargo,foregone_harvest=float(gross),
                stay_toll=float(was),moving_toll=float(now),old_cell=p,new_cell=z,nominal_saving=float(-n))
