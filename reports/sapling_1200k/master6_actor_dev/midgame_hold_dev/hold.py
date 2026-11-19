"""Hold an occupied non-base lane while a credible loaded opponent can still approach it."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'midgame_lane_dev'))
from route import propose,apply,_TO,_D,_DIRECTION
def defend(obs,orders,root):
    seat=obs['player'];own,enemy=obs['players'][seat],obs['players'][1-seat]
    guard=own[2].get(root['worker']);target=enemy[2].get(root['target'])
    if not guard or not target or target[1]<=guard[1] or guard[0]!=root['block_cell']:return orders
    pos,load,_=guard;carrier,_,_=target
    if _D[pos,carrier]>3:return orders
    if not any(_D[carrier,b]==_D[carrier,pos]+_D[pos,b] for b in enemy[1].values()):return orders
    if any(w[2]=='COLLECTOR' and w[1]<=load and _D[pos,w[0]]<=1 for w in enemy[2].values()):return orders
    if any(k!=root['worker'] and _TO[w[0],_DIRECTION.index(orders.get(k))]==pos for k,w in own[2].items()):return orders
    result=dict(orders);result.pop(root['worker'],None)
    return result
