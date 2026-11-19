"""Apply the additional public arrival-time cap without refitting the original five-step predictor."""
import json
from scan import HERE,ACTOR,ROOT,FastCarbon,read,sha,simulate,defend,DIST
from forecast import vector,predict
from fit import scan_tape,frozen_source


def match(source,model):
    tape=frozen_source(source);s=source['seat'];cfg=tape['game_configuration']
    sim=FastCarbon(cfg).reset(seed=source['seed'])
    game={'seat':s,'tape':tape['actions'],'cfg':cfg}
    active={};history=[];first=None;alarms=0
    for t,joint in enumerate(game['tape']):
        obs=sim.obs(s); own,rival=obs['players'][s],obs['players'][1-s];anyone={tid for pl in obs['players'] for tid in pl[3]}
        history.append({u:v[0] for u,v in rival[2].items() if v[2]=='PLANTER'});history=history[-4:]
        options=[]
        for tid,tree in active.items():
            if t+10>len(game['tape']) or cfg['treeLifespan']-own[3][tid][1]<10:continue
            p=tree['pos'];risk=predict(model,vector(obs,tree,history,cfg))
            if risk<.35:continue
            deadline=min([5]+[DIST[p][v[0]]+1 for v in rival[2].values() if v[2]=='PLANTER']+
                         [DIST[p][pos]+2 for pos in rival[1].values()])
            candidates=sorted((v[1],DIST[p][v[0]],uid) for uid,v in own[2].items()
                              if v[2]=='COLLECTOR' and v[1]<=25 and DIST[p][v[0]]<=deadline)
            if not candidates:continue
            cargo,d,uid=candidates[0];cmd,why=defend(obs,joint[s],uid,p)
            if cmd is None or cmd==joint[s].get(uid,'NONE'):continue
            alarms+=1
            options.append((-risk,tid,uid,cargo,d,deadline,cmd,tree))
        if options and first is None:
            risk,tid,uid,cargo,d,lim,cmd,tree=min(options); snap=sim.get_state(); target={**tree,'end':t+5}
            base=simulate(game,target,5,snap,uid,False);alt=simulate(game,target,5,snap,uid,True)
            first=dict(root=t,tid=tid,uid=uid,risk=-risk,distance=d,deadline=lim,cargo=cargo,base=base,alternative=alt,
                       delta_margin=round(alt['cash'][s]-alt['cash'][1-s]-base['cash'][s]+base['cash'][1-s],2))
        sim.step(*joint);post=sim._players_obs()[s][3]
        for tid in list(active):
            if tid not in post:del active[tid]
        if 51<=t+1<=250:
            for tid in set(post)-anyone:active[tid]={'start':t+1,'tid':tid,'pos':post[tid][0]}
    if list(sim.cash)!=source['scores']:raise ValueError('original terminal mismatch')
    return {'eligible':alarms,'first':first}


def main():
    prereg=HERE/'race_protocol.json';m=read(HERE/'warning_weights.json')['models']
    source=ACTOR/'tree_response_training/supplement_01/batches/intake_06/source_receipt.json'
    rows=[]
    for game in read(source)['games']:
        _,_,original=scan_tape(game,m)
        rows.append(dict(seed=game['seed'],seat=game['seat'],source_sha256=game['raw_sha256'],
                         original_selector=original,arrival_constrained=match(game,m['public'])))
    if len(rows)!=4 or len({g['seed'] for g in rows})!=2:raise ValueError('incomplete two-seat panel')
    out={'protocol_sha256':sha(prereg),'source_receipt_sha256':sha(source),'warning_weights_sha256':sha(HERE/'warning_weights.json'),
         'simulation_sha256':sha(HERE/'scan.py'),'code_sha256':sha(HERE/'race.py'),'games':rows}
    (HERE/'race_transfer.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
