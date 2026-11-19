from contact import _TO,_D,_DIRECTION
def harvest(obs,orders,r):
    s=obs['player'];own,enemy=obs['players'][s],obs['players'][1-s]
    worker=own[2].get(r['worker'])
    if not worker:return orders
    p,load,kind=worker
    if p!=r['site'] or (obs['carbon'][p] or 0)<15:return orders
    if any(w[2]=='COLLECTOR' and w[1]<=load and _D[w[0],p]<=1 for w in enemy[2].values()):return orders
    if any(_TO[w[0],_DIRECTION.index(orders.get(k))]==p for k,w in own[2].items() if k!=r['worker']):return orders
    if any(a[0]==p for v in obs['players'] for a in v[3].values()):return orders
    new=dict(orders);new.pop(r['worker'],None);return new
