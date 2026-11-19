"""Public features for an approaching freight carrier; moves are in physical board coordinates."""
import numpy as np

_DIRECTION = [None, "UP", "RIGHT", "DOWN", "LEFT"]
_axis = np.arange(225)
_y, _x = divmod(_axis,15)
_D = np.minimum(abs(_y[:,None]-_y),15-abs(_y[:,None]-_y)) + np.minimum(abs(_x[:,None]-_x),15-abs(_x[:,None]-_x))
_TO = np.stack([_axis,((_y-1)%15)*15+_x,_y*15+(_x+1)%15,((_y+1)%15)*15+_x,_y*15+(_x-1)%15],1)

def carriers(players, seat):
    own, opp = players[seat],players[1-seat]
    bases = list(own[1].values())
    if not bases:return
    for uid,(pos,load,kind) in own[2].items():
        if kind!="COLLECTOR" or load<80:continue
        home=min(_D[pos,b] for b in bases)
        if home>5:continue
        robbers=[v[0] for v in opp[2].values() if v[2]=="COLLECTOR" and v[1]<load and _D[pos,v[0]]<=3]
        if not robbers:continue
        dest=_TO[pos]
        dist_home=_D[np.ix_(dest,bases)].min(1)
        danger=_D[np.ix_(dest,robbers)].min(1)
        team=[w[0] for k,w in own[2].items() if k!=uid and _D[pos,w[0]]<=2]
        congest=np.isin(dest,team)
        stay=np.arange(5)==0; toward=home-dist_home;bank=dist_home==0
        features=np.stack([stay,toward,bank,danger==0,danger==1,np.minimum(danger,3),congest,
                           stay*min(load/300,2),toward*min(load/300,2),bank*(danger<=1)],axis=1)
        yield uid,load,dest,features

def cargo_map(observation,weights):
    field=np.zeros(225)
    for _,load,locations,features in carriers(observation["players"],1-observation["player"]):
        logits=features@weights
        p=np.exp(logits-logits.max())
        field[locations]+=p/p.sum()*min(load/200,1.)
    return np.minimum(field,1.)
