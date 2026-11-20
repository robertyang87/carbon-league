"""Collect root-legal single-collector opportunities in the last twenty decisions of the rematch."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
LADDER=HERE.parents[1]
sys.path.insert(0,str(LADDER/'v4_alphacarbon_decisions'))
from scan import gzip,json,copy,hashlib,rule,after,move,dist,ACTIONS,feasible,own_capture,new_order,act_history
SOURCE=LADDER/'v4_observations/increment_0926/replays/5906206b61c94c29d8ea2f99d2b4d62d-1.json.gz'
def examine():
    data=SOURCE.read_bytes();game=json.loads(gzip.decompress(data));f=game['steps']
    assert game['rewards']==[6245.53,6493.12]
    sim=rule.econ.FastCarbon(game['configuration']).reset(full_carbon=f[0][0]['observation']['full_carbon']);sim.cash=rule.econ.CashEvents(sim.cash)
    selected=[];n=0;savings=[]
    for t in range(299):
        p=f[t][0]['observation']['players'];actions=[r['action'] or {} for r in f[t+1]]
        if 279<=t<=298:
            _,base=after(sim,actions)
            for uid,data in p[0][2].items():
                if data[2]!='COLLECTOR':continue
                for a in ACTIONS:
                    if a==actions[0].get(uid,'NONE') or not feasible(p,actions[0],uid,a):continue
                    n+=1;pair=[new_order(actions[0],uid,a),actions[1]]
                    s,r=after(sim,pair)
                    caught={k:v for k,v in own_capture(r).items() if k not in own_capture(base)}
                    if caught or r['bank'][0]>base['bank'][0] or r['bank'][1]<base['bank'][1]:
                        selected.append(dict(step=t,uid=uid,start=data,original=actions[0].get(uid,'NONE'),alternative=a,
                               captured=caught,bank_change=[q-z for q,z in zip(r['bank'],base['bank'])],
                               cash_gain=round(r['margin']-base['margin'],2)))
        sim.cash.events.clear();sim.step(*actions)
        assert sim.obs(0)['players']==f[t+1][0]['observation']['players'] and list(sim.carbon)==f[t+1][0]['observation']['full_carbon']
    return dict(input_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),n=n,options=selected)
if __name__=='__main__':
    j=examine();(HERE/'scan.json').write_text(json.dumps(j,indent=2)+'\n')
    for x in j['options']:print(x)
