"""Map-equal warning model; evaluate an executable warning including false alarms."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from scan import HERE,ACTOR,ROOT,ECO,MOVES,DIST,FastCarbon,defend,read,sha,simulate,sources
from forecast import FEATURES,vector,predict


def clean(c):
    return json.loads(json.dumps(c,allow_nan=False))


def frozen_source(source):
    raw=ROOT/source['raw_path'];tape=read(raw)
    if sha(raw)!=source['raw_sha256']:raise ValueError('source identity')
    return tape


def scan_tape(source, models=None):
    tape=frozen_source(source);s=source['seat']
    game={'seat':s,'tape':tape['actions'],'cfg':tape['game_configuration']}
    sim=FastCarbon(game['cfg']).reset(seed=source['seed'])
    active={}; root_rows=[]; selected=None; exposed=0; history=[]; cases=[]
    for t,joint in enumerate(game['tape']):
        obs=sim.obs(s); pre=obs['players'][s][3]; anyone={tid for v in obs['players'] for tid in v[3]}
        history.append({u:v[0] for u,v in obs['players'][1-s][2].items() if v[2]=='PLANTER'})
        history=history[-4:]
        rank=[]
        for tid,tree in active.items():
            if t+5>len(game['tape']):continue
            x=vector(obs,tree,history,game['cfg'])
            root_rows.append((source['seed'],t,tid,x))
            if not models or t+10>len(game['tape']) or x[1]<10:continue
            pred=predict(models['public'],x)
            if pred<.35:continue
            choices=sorted((cargo,DIST[pos][tree['pos']],uid) for uid,(pos,cargo,kind) in obs['players'][s][2].items()
                           if kind=='COLLECTOR' and cargo<=25 and DIST[pos][tree['pos']]<=5)
            if not choices:continue
            cargo,dist,uid=choices[0];cmd,reason=defend(obs,joint[s],uid,tree['pos'])
            if cmd is None or cmd==joint[s].get(uid,'NONE'):continue
            exposed+=1
            rank.append((-pred,tid,uid,cmd,tree))
        if rank and selected is None:
            q,tid,uid,cmd,tree=min(rank)
            snap=sim.get_state(); target={**tree,'end':t+5}
            arms={k:simulate(game,target,5,snap,uid,k=='defense') for k in ('parent','defense')}
            parent,candidate=arms.values()
            selected={'root':t,'tid':tid,'start':tree['start'],'uid':uid,'cargo_at_entry':obs['players'][s][2][uid][1],
                      'p':-q,'parent_order':joint[s].get(uid,'NONE'),'alternative':cmd,'outcomes':arms,
                      'cash_margin_delta':round(candidate['cash'][s]-candidate['cash'][1-s]-parent['cash'][s]+parent['cash'][1-s],2),
                      'focal_delivery_delta':candidate['focal_deposits']-parent['focal_deposits'],
                      'focal_cargo_delta':candidate['focal_cargo']-parent['focal_cargo']}
        sim.step(*joint)
        own=sim._players_obs()[s][3]
        for tid in list(active):
            if tid not in own:
                r=active.pop(tid);cases.append({**r,'end':t+1,'taken':tid in sim.trees})
        if 51<=t+1<=250:
            for tid in set(own)-anyone:
                active[tid]={'tid':tid,'pos':own[tid][0],'start':t+1}
    cases.extend({**r,'end':len(game['tape']),'taken':False} for r in active.values())
    if list(sim.cash)!=source['scores']:raise ValueError('official terminal mismatch')
    losses={r['tid']:r['end'] for r in cases if r['taken']}
    labels=[int(t<losses.get(tid,10**6)<=t+5) for _,t,tid,x in root_rows]
    if selected is not None:
        selected['actually_seized_in_5']=int(selected['root']<losses.get(selected['tid'],10**6)<=selected['root']+5)
    return root_rows,labels,{'seed':source['seed'],'seat':s,'raw_sha256':source['raw_sha256'],'trigger_opportunities':exposed,
                             'emitted':selected}


def logistic(x,y,weight,cols):
    a=x[:,cols];mu=(weight[:,None]*a).sum(0);sd=np.maximum(.01,np.sqrt((weight[:,None]*(a-mu)**2).sum(0)))
    X=np.column_stack([np.ones(len(a)),(a-mu)/sd]);beta=np.zeros(X.shape[1])
    ridge=np.diag([0]+[.004]*(X.shape[1]-1))
    for _ in range(40):
        p=1/(1+np.exp(-np.clip(X@beta,-35,35)))
        grad=X.T@(weight*(p-y))+ridge@beta
        hess=X.T@((weight*p*(1-p))[:,None]*X)+ridge
        update=np.linalg.solve(hess,grad);beta-=update
        if max(abs(update))<1e-8:break
    return dict(columns=cols,center=mu.tolist(),scale=sd.tolist(),weights=beta[1:].tolist(),bias=beta[0])


def main():
    split={};raw={};inputs={}
    pools={'train':[],'late_training_maps':[],'later_development':[]}
    for label,s in sources():
        part='later_development' if label=='held_development' else ('train' if s['seed']<621474000 else 'late_training_maps')
        pools[part].append(s)
    rows=[];ys=[]
    for s in pools['train']:
        r,y,_=scan_tape(s);rows+=r;ys+=y
    gids=[r[0] for r in rows];counts=Counter(gids);w=np.array([1/(len(counts)*counts[g]) for g in gids])
    X=np.asarray([r[3] for r in rows]);y=np.array(ys)
    models={'public':logistic(X,y,w,list(range(len(FEATURES)))),'distance_life':logistic(X,y,w,[0,1])}
    outfile=HERE/'warning_weights.json'
    payload=dict(features=FEATURES,models=models,source_sha256=[s['raw_sha256'] for s in pools['train']],roots=len(y),positives=int(y.sum()),
                 protocol_sha256=sha(HERE/'prediction_protocol.json'),code_sha256=sha(Path(__file__)),forecast_sha256=sha(HERE/'forecast.py'))
    outfile.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    for part in ('late_training_maps','later_development'):
        tests=[]; forecasts={}
        for s in pools[part]:
            r,actual,played=scan_tape(s,models)
            predicted={k:np.asarray([predict(net,x[3]) for x in r]) for k,net in models.items()}
            y=np.array(actual,dtype=int); tests.append(played)
            item=forecasts.setdefault(s['seed'],{'n':0,'positive':0,'err':Counter(),'alarm':0,'true_alarm':0})
            item['n']+=len(y);item['positive']+=int(y.sum())
            for k,p in predicted.items():item['err'][k]+=float(((p-y)**2).sum())
            alarm=predicted['public']>=.35;item['alarm']+=int(alarm.sum());item['true_alarm']+=int(y[alarm].sum())
        for vals in forecasts.values():vals['brier']={k:v/vals['n'] for k,v in vals.pop('err').items()}
        raw[part]={'per_map':forecasts,'brier_map_equal':{k:np.mean([v['brier'][k] for v in forecasts.values()]) for k in models},'games':tests}
    (HERE/'warning_transfer.json').write_text(json.dumps({'weights_sha256':sha(outfile),'panels':raw},indent=2,allow_nan=False)+'\n')
    print([(part,v['brier_map_equal']) for part,v in raw.items()],flush=True)


if __name__=='__main__':main()
